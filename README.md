
<!-- <div align="center"> -->



# LiteFocus 

<!-- <img src="assets/LOGO.jpg" height="128px" style="border-radius: 28px;"/> -->
<!-- <br> -->
<a href="https://arxiv.org/abs/2407.10468"><img src="https://img.shields.io/badge/ariXv-2407.10468-A42C25.svg" alt="arXiv"></a>
<br>

<!-- </div> -->

> **Video-Infinity: Distributed Long Video Generation**
> <br>
> Zhenxiong Tan, 
> [Xinyin Ma](https://horseee.github.io), 
> [Gongfan Fang](https://fangggf.github.io), 
> and 
> [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)
> <br>
> [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore
> <br>


## TL;DR (Too Long; Didn't Read)
LiteFocus is a tool designed to accelerate diffusion-based TTA model, now implemented with the base model [AudioLDM2](https://audioldm.github.io/audioldm2). It doubles the processing speed and enhances audio quality.


## Setup
* **Prepare Environment (optional)**
```bash
conda create -n litefocus python=3.10
conda activate litefocus
```
* **Install Base Model**
```bash
pip3 install git+https://github.com/haoheliu/AudioLDM2.git
```


## Usage
### Basic Usage
```diff
from audioldm2 import text_to_audio, build_model
import scipy

+ from litefocus import inject_lite_focus, disable_lite_focus

model = build_model(model_name='audioldm2-full')

+ inject_lite_focus(model)

waveform = text_to_audio(
    latent_diffusion=model,
    duration=40,
    text='Musical constellations twinkling in the night sky, forming a cosmic melody.',
)

scipy.io.wavfile.write("out.wav", rate=16000, data=waveform)
```

### Disable LiteFocus
```python
disable_lite_focus(model)
```


### Configuration
```python
config = {
    'same_frequency': True,
    'cross_frequency': True,
    'sparse_ratio': 0.1
}

inject_lite_focus(model, config)
```


| Parameter         | Description                                                            | Default Value |
| ----------------- | ---------------------------------------------------------------------- | ------------- |
| `same_frequency`  | Enables attention to tokens sharing the same-frequency.                | `True`        |
| `cross_frequency` | Enables attention to tokens in cross-frequency           compensation. | `True`        |
| `sparse_ratio`    | Specifies the sparsity ratio for `cross_frequency`.                    | 0.1           |


## To-Do
- [x] AudioLDM2 Integration
- [ ] Diffusers pipeline Integration

## Citation
```
@article{
  tan2024lite,
  title={LiteFocus: Accelerated Diffusion Inference for Long Audio Synthesis},
  author={Zhenxiong Tan, Xinyin Ma, Gongfan Fang, and Xinchao Wang},
  journal={arXiv preprint arXiv:2407.10468},
  year={2024}
}
```
