# Changes


### Input shape and ratio flexibility.

1. Input images will not be reshaped to a specific image size. Training image ratios are in tha range: `[0.5, 2.5]`, thus if your image ratio is outside of these boundaries, padding will be applied.

2. UnidepthV2 exposes the attribute `self.resolution_level` (with range `[0,10)`) that is used in the preprocess function and can be used to tradeoff resolution and speed, with **ideally small effect** on the output scale. In particular, the level describes the linear interpolation degree of the processed image area within the training bounds. The training image area (named "pixels") for ViT are in the range `[0.2, 0.6]` MegaPixels (see `pixels_bounds` in config). If no attribute is set, the full pixel bounds are used.

3. Infer method will use interpolation mode defined by the attribute `self.interpolation_mode`, default is `bilinear`.


### Confidence output  

The model outputs confidence as the estimated scale-invaraint log error, i.e. the confidence is a ranking and relative within one input.
In particular, it does not have an absolute meaning (e.g. no heteroschdastic noise modelling).


### Faster inference  

The model is >30% faster than V1, tested on RTX4090 with float16 data-type.


### Random edge extraction  

In order to perform `EdgeGuidedLocalSSI` efficiently, you need to compile the CUDA operation we wrote to speedup the random patch extraction with `cd ./unidepth/ops/extract_patches && bash compile.sh`. Remember to export your `TORCH_CUDA_ARCH_LIST` if it is different wrt the defualt present in `compile.sh`.


### Camera

UniDepthV2 can accept a larger variety of cameras, plase check `unidepth/utils/camera.py` for a comprehensive list.
`infer` method accepts either a tensor as a K matrix, i.e. assumes  pinhole!, or a **child** of `Camera`, which can be, e.g. Fisheye624, Pinhole, OPENCV, ...
Do not pass directly a `Camera` class as it lacks methods and it is supposed to be an abtract parent class with some methods implemented.


### ONNX support

We added support to UniDepthV2 in __ONNX__ format.
For instance you can run from the root of the repo:
```bash

python ./unidepth/models/unidepthv2/export.py --version v2 --backbone vitl --shape 462 616 --output-path unidepthv2.onnx
```

Shape will be changed to the closest shape which is multiple of 14, i.e. ViT patch size.
Your input shape at inference time will have to match with the (resized) shape passed to the exporter!
The corresponding __ONNX__ model does not do any pre- or post-processing.
Therefore, you should input an ImageNet-statistic normalized rgb image rescaled to the given input shape and.
Add falg `--with-camera` to accept given GT camera as unprojected rays as input.