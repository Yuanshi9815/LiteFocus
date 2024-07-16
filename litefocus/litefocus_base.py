import torch
import math
from einops import rearrange

def attention_lite_focus(
        query, key, value, 
        scale=None,
        config={}
        ):
    config = {
        **{
            'same_frequency': True,
            'cross_frequency': True,
            'sparse_ratio': 0.1,
            'f_n': 8,
        },
        **config,
    }

    f_n = config['f_n']

    heads_n, query_token_n, feature_n = query.size(-3), query.size(-2), query.size(-1)
    time_n = query_token_n // f_n
    batch_n = query.size(0)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    final_attn_weights = []

    # 2. compute same freqency attention weights
    if config['same_frequency']:
        query_ = query.view([-1, heads_n, time_n, f_n, feature_n])#.transpose(2,3)
        key_ = key.view([-1, heads_n, time_n, f_n, feature_n])#.transpose(2,3)
        value_2 = value.view([-1, heads_n, time_n, f_n, feature_n])#.transpose(2,3)
        
        attn_weight_2 = torch.einsum(
            'bhtfc,bhyfc -> bhtfy', 
            query_, 
            key_
        ) * scale_factor
        attn_weight_2 = attn_weight_2.reshape([batch_n, heads_n, time_n * f_n, -1])
        final_attn_weights.append(attn_weight_2)

    # 3. random global attention
    if config['cross_frequency']:
        sparse_ratio = config['sparse_ratio']
        # 1. get random indices
        indices = torch.randperm(time_n * f_n)
        indices = indices[:int(time_n * f_n * sparse_ratio)]
        # 2. get attention weights
        key_ = key[:, :, indices, :]
        value_3 = value[:, :, indices, :]
        attn_weight_3 = query @ key_.transpose(-2, -1) * scale_factor
        final_attn_weights.append(attn_weight_3)

    if final_attn_weights:
        final_attn_weights = torch.cat(final_attn_weights, dim=-1)
        final_attn_weights = torch.softmax(final_attn_weights, dim=-1)

    point = 0
    out_result = None

    if config['same_frequency']:
        # get softmaxed attention weights
        k_n_2 = attn_weight_2.shape[-1]
        attn_weight_2 = final_attn_weights[:, :, :, point:point+k_n_2].reshape([batch_n, heads_n, time_n, f_n, -1])
        point += k_n_2
        # compute result
        result = torch.einsum(
            'bhtfy,bhyfc -> bhtfc', 
            attn_weight_2, 
            value_2
        ).reshape([batch_n, heads_n, time_n * f_n, -1])
        out_result = result if out_result is None else out_result + result

    if config['cross_frequency']:
        # get softmaxed attention weights
        k_n_3 = attn_weight_3.shape[-1]
        attn_weight_3 = final_attn_weights[:, :, :, point:point+k_n_3]
        point += k_n_3
        # compute result
        result = attn_weight_3 @ value_3
        out_result = result if out_result is None else out_result + result

    return out_result


def get_new_forward(module, config={}):
    self = module
    def new_forward(x, context=None, mask=None):
        if context is not None:
            return self.old_forward(x, context, mask)
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        out = attention_lite_focus(q, k, v, 
                           scale=self.scale, 
                           config=config,
                        )

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)
    
    return new_forward


def inject_lite_focus(audioldm_model, config={}):
    attention_modules = []
    for layer_id in [4,5]:
        for module_id in [1,2,3]:
            transformer_module = audioldm_model.model.diffusion_model.input_blocks[layer_id][module_id].transformer_blocks[0]
            attention_modules.append(transformer_module.attn1)
            attention_modules.append(transformer_module.attn2)
    for layer_id in [6,7,8]:
        for module_id in [1,2,3]:
            transformer_module = audioldm_model.model.diffusion_model.output_blocks[layer_id][module_id].transformer_blocks[0]
            attention_modules.append(transformer_module.attn1)
            attention_modules.append(transformer_module.attn2)
    for module in attention_modules:
        if not hasattr(module, 'old_forward'):
            module.old_forward = module.forward
        module.forward = get_new_forward(module, config)


def disable_lite_focus(audioldm_model):
    attention_modules = []
    for layer_id in [4,5]:
        for module_id in [1,2,3]:
            transformer_module = audioldm_model.model.diffusion_model.input_blocks[layer_id][module_id].transformer_blocks[0]
            attention_modules.append(transformer_module.attn1)
            attention_modules.append(transformer_module.attn2)
    for layer_id in [6,7,8]:
        for module_id in [1,2,3]:
            transformer_module = audioldm_model.model.diffusion_model.output_blocks[layer_id][module_id].transformer_blocks[0]
            attention_modules.append(transformer_module.attn1)
            attention_modules.append(transformer_module.attn2)
    for module in attention_modules:
        if not hasattr(module, 'old_forward'):
            continue
        module.forward = module.old_forward
