# Changes


### Input shape and ratio flexibility.

1. Input images will not be reshaped to a specific image size. Training image ratios are in tha range: `[2/3, 2/1]`, thus if your image ratio is outside of these boundaries, we suggest to crop or pad it to be within the image ratio bounds.

2. UnidepthV2 exposes the attribute `self.resolution_level` (with range `[0,10]`) that is used in the preprocess function and can be used to tradeoff resolution and speed, with **possible effect** on the output scale. In particular, the level describes the linear interpolation degree of the processed image area within the training bounds. The training image area (named "pixels") for ViT are in the range `[1400, 2400]` (see `pixels_bounds` in config). If no attribute is set, the max level, i.e. 10, will be used. We improperly use the concept of "pixels" which accounts for the image area after patchification, e.g. for ViT means that it is `1/14**2` the actual original image area.

3. Infer method will use interpolation mode defined by the attribute `self.interpolation_mode`, default is `bilinear`.


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