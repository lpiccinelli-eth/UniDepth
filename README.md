[![arXiv](https://img.shields.io/badge/arXiv-UniDepth-red)](https://arxiv.org/abs/2403.18913)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Cooming%20Soon-yellow)](https://huggingface.co/spaces/lpiccinelli/UniDepth)

[![KITTI Benchmark](https://img.shields.io/badge/KITTI%20Benchmark-1st%20(at%20submission%20time)-blue)](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unidepth-universal-monocular-metric-depth/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=unidepth-universal-monocular-metric-depth)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unidepth-universal-monocular-metric-depth/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=unidepth-universal-monocular-metric-depth)


# UniDepth: Universal Monocular Metric Depth Estimation

![](assets/docs/unidepth-banner.png)

> [**UniDepth: Universal Monocular Metric Depth Estimation**](https://arxiv.org/abs/2403.18913),  
> Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, Fisher Yu,  
> CVPR 2024 (to appear),  
> *Paper at [arXiv 2403.18913](https://arxiv.org/pdf/2403.18913.pdf)*  


## News and ToDo

- [ ] Release UniDepth on PyPI.
- [ ] Release HuggingFace/Gradio demo.
- [ ] Release smaller V2 models (Small and Base).
- [x] `01.05.2024`: Release UniDepthV2.
- [x] `02.04.2024`: Release UniDepth as python package.
- [x] `01.04.2024`: Inference code and V1 models are released.
- [x] `26.02.2024`: UniDepth is accepted at CVPR 2024! (Highlight :star:)


## Zero-Shot Visualization

### YouTube (The Office - Parkour)
<p align="center">
  <img src="assets/docs/theoffice.gif" alt="animated" />
</p>

### NuScenes (stitched cameras)
<p align="center">
  <img src="assets/docs/nuscenes_surround.gif" alt="animated" />
</p>


## Installation

Requirements are not in principle hard requirements, but there might be some differences (not tested):
- Linux
- Python 3.10+ 
- CUDA 11.8

Install the environment needed to run UniDepth with:
```shell
export VENV_DIR=<YOUR-VENVS-DIR>
export NAME=Unidepth

python -m venv $VENV_DIR/$NAME
source $VENV_DIR/$NAME/bin/activate

# Install UniDepth and dependencies
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

# Install Pillow-SIMD (Optional)
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

If you use conda, you should change the following: 
```shell
python -m venv $VENV_DIR/$NAME -> conda create -n $NAME python=3.11
source $VENV_DIR/$NAME/bin/activate -> conda activate $NAME
```

*Note*: Make sure that your compilation CUDA version and runtime CUDA version match.  
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

*Note*: xFormers may raise the the Runtime "error": `Triton Error [CUDA]: device kernel image is invalid`.  
This is related to xFormers mismatching system-wide CUDA and CUDA shipped with torch.  
It may considerably slow down inference.

Run UniDepth on the given assets to test your installation (you can check this script as guideline for further usage):
```shell
python ./scripts/demo.py
```
If everything runs correctly, `demo.py` should print: `ARel: 5.13%`.

If you encounter `Segmentation Fault` after running the demo, you may need to uninstall torch via pip (`pip uninstall torch`) and install the torch version present in [requirements](requirements.txt) with `conda`.

## Get Started

After installing the dependencies, you can load the pre-trained models easily from [Hugging Face](https://huggingface.co/models?other=UniDepth) as follows:

```python
from unidepth.models import UniDepthV1

model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14") # or "lpiccinelli/unidepth-v1-cnvnxtl" for the ConvNext backbone
```

Then you can generate the metric depth estimation and intrinsics prediction directly from RGB image only as follows:

```python
import numpy as np
from PIL import Image

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the RGB image and the normalization will be taken care of by the model
rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W

predictions = model.infer(rgb)

# Metric Depth Estimation
depth = predictions["depth"]

# Point Cloud in Camera Coordinate
xyz = predictions["points"]

# Intrinsics Prediction
intrinsics = predictions["intrinsics"]
```

You can use ground truth intrinsics as input to the model as well:
```python
intrinsics_path = "assets/demo/intrinsics.npy"

# Load the intrinsics if available
intrinsics = torch.from_numpy(np.load(intrinsics_path)) # 3 x 3

predictions = model.infer(rgb, intrinsics)
```

To use the forward method for your custom training, you should:  
1) Take care of the dataloading:  
  a) ImageNet-normalization  
  b) Long-edge based resizing (and padding) with input shape provided in `image_shape` under configs  
  c) `BxCxHxW` format  
  d) If any intriniscs given, adapt them accordingly to your resizing  
2) Format the input data structure as:  
```python
data = {"image": rgb, "K": intrinsics}
predictions = model(data, {})
```

Please visit [Hugging Face](https://huggingface.co/lpiccinelli) to access the repo models with weights. You can load UniDepth as:

```python
from unidepth.models import UniDepthV1, UniDepthV2

model_v1 = UniDepthV1.from_pretrained(f"lpiccinelli/unidepth-v1-{backbone}")
model_v2 = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-{backbone}")
```

In addition, we provide loading from TorchHub as:

```python
version = "v2"
backbone = "vitl14"

model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True, trust_repo=True, force_reload=True)
```

where backbones available are "vitl14" and "cnvnxtl", and versions available are "v1" and "v2".
You can look into function `UniDepth` in [hubconf.py](hubconf.py) to see how to instantiate the model from local file: provide a local `path` in line 34.


## UniDepthV2

Visit [UniDepthV2 ReadMe](assets/docs/V2_README.md) for a more detailed changelog.
To summarize the main differences are:  
- Input shape and ratio flexibility.  
- Confidence output  
- Decoder design  
- Faster inference  
- ONNX support


## Results

### Metric Depth Estimation
The performance reported is for UniDepthV1 model and the metrics is d1 (higher is better) on zero-shot evaluation. The common split between SUN-RGBD and NYUv2 is removed from SUN-RGBD validation set for evaluation. 
*: non zero-shot on NYUv2 and KITTI.

| Model | NYUv2 | SUN-RGBD | ETH3D | Diode (In) | IBims-1 | KITTI | Nuscenes | DDAD | 
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| BTS* | 88.5 | 76.1 | 26.8 | 19.2 | 53.1 | 96.2 | 33.7 | 43.0 |
| AdaBins* | 90.1 | 77.7 | 24.3 | 17.4 | 55.0 | 96.3 | 33.3 | 37.7 |
| NeWCRF* | 92.1 | 75.3 | 35.7 | 20.1 | 53.6 | 97.5 | 44.2 | 45.6 | 
| iDisc* | 93.8 | 83.7 | 35.6 | 23.8 | 48.9 | 97.5 | 39.4 | 28.4 |
| ZoeDepth* | 95.2 | 86.7 | 35.0 | 36.9 | 58.0 | 96.5 | 28.3 | 27.2 |
| Metric3D | 92.6 | 15.4 | 45.6 | 39.2 | 79.7 | 97.5 | 72.3 | - |
| UniDepth_ConvNext | 97.2| 94.8 | 49.8 | 60.2 | 79.7 | 97.2 | 83.3 | 83.2 |
| UniDepth_ViT | 98.4 | 96.6 | 32.6 | 77.1 | 23.9 | 98.6 | 86.2 | 86.4 |


## Contributions

If you find any bug in the code, please report to Luigi Piccinelli (lpiccinelli@ethz.ch)


## Citation

If you find our work useful in your research please consider citing our publication:
```bibtex
@inproceedings{piccinelli2024unidepth,
    title={UniDepth: Universal Monocular Metric Depth Estimation},
    author = {Piccinelli, Luigi and Yang, Yung-Hsu and Sakaridis, Christos and Segu, Mattia and Li, Siyuan and Van Gool, Luc and Yu, Fisher},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```


## License

This software is released under Creatives Common BY-NC 4.0 license. You can view a license summary [here](LICENSE).


## Acknowledgement

We would like to express our gratitude to [@niels](https://huggingface.co/nielsr) for helping intergrating UniDepth in HuggingFace.

This work is funded by Toyota Motor Europe via the research project [TRACE-Zurich](https://trace.ethz.ch) (Toyota Research on Automated Cars Europe).
