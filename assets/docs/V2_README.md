# Changes


### Input shape and ratio flexibility.

Input images do not have to be resized to a specific ratio. Training ratios are in tha range: `[2/3, 2/1]`. The image shape and size is flexible and can be adapted to the needs and performances desired. "Pixels" are meant after the patchification, for instance for ViT they are `(1/14)**2` of the original input number of pixels. Training number of pixels for ViT are in the range `[1800, 2400]` as in the config. The model will ue the training values and try to find the closest input size as `max(min(num_pixels, pixels_bounds[1]), pixels_bounds[0])` (line 51 in `unidepthv2.py`). You can modify it by changing the attribute `self.shape_constraints["pixels_bound"]` in `UniDepthV2` object. For instance, to use the largest training input size you can set `self.shape_constraints["pixels_bound"] = [2400, 2400]`. If you set less than the maximum a warning is triggered. For instance, you can use the maximum as:

```python
from unidepth.models import UniDepthV2

model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
max_pixels = model.shape_constraints["pixels_bound"][1]
model.shape_constraints["pixels_bound"] = [max_pixels, max_pixels]
```

Infer method will use interpolation mode expressed by the attribute `self.interpolation_mode`, default is `nearest-exact`.


### Confidence output  

The model outputs confidence in the range `[0, 1]` and represent the ARel error after affine matching with GT. The confidence itself is shift invariance, namely the confidence is a ranking and relative within one input. In particular, it does not have an absolute meaning (e.g. no heteroschdastic noise modelling).


### Decoder design predicting separately scale-shift invariant depth and scale and shift to allow more diverse training. 

The decoder presents three heads: `Camera`, `Depth` and `Global`. `Depth` head predicts scale and shift invariant depth: exponential of normalized values.
`Global` head predicts the scale and shift to match the `Depth` head output to metric.
With such design we can mix seamlessly dataset with metric GT, scale-invariant (i.e., SfM) or scale-shift invariant by turning down the gradient to the `Global` head when GT is either scale or shift invariant.
This allows to scale up the training variety.
Version 1 and 2 present similar performance but output of version 2 may look more nervous because more diversity is linked to lower GT quality, thus introducing artifacts... 


### Faster inference  

The model is >30% faster than V1, tested on RTX4090 with float16 data-type.


### ONNX support

We added support to UniDepthV2 in __ONNX__ format.
Both with and without gt intrinsics support.
It does not allow for dynamic shapes at test time.
For instance you can run from the root of the repo:
```bash

python ./unidepth/models/unidepthv2/export.py --version v2 --backbone vitl14 --shape (462, 616) --output-path unidepthv2.onnx --with-camera
```

Shape will be changed to the closest shape which is multiple of 14, i.e. ViT patch size.
Your input shape at inference time will have to match with the (resized) shape passed to the exporter!
The corresponding __ONNX__ model does not do any pre- or post-processing.
Therefore, you should input an ImageNet-statistic normalized rgb image rescaled to the given input shape and, if `--with-camera` the corresponding (properly rescaled) camera intrinsics, too.


Disclaimer: Not fully tested