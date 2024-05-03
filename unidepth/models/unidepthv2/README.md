# Changes


### Input shape and ratio flexibility.

Input images do not have to be resized to a specific ratio. Training ratios are in tha range: `[2/3, 2/1]`. The image shape and size is flexible and can be adapted to the needs and performances desired. "Pixels" are meant after the patchification, for instance for ViT they are `(1/14)**2` of the original input number of pixels. Training number of pixels for ViT are in the range `[1800, 2400]` as in the config. The model will ue the training values and try to find the closest input size as `max(min(num_pixels, pixels_bounds[1]), pixels_bounds[0])` (line 51 in `unidepthv2.py`). You can modify it by changing the attribute `self.shape_constraints["pixels_bound"]` in `UniDepthV2` object. For instance, to use the largest training input size you can set `self.shape_constraints["pixels_bound"] = [2400, 2400]`. If you set less than the maximum a warning is triggered. For instance, you can use the maximum as:

```python
from unidepth.models import UniDepthV2

model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
max_pixels = model.shape_constraints["pixels_bound"][1]
model.shape_constraints["pixels_bound"] = [max_pixels, max_pixels]
```


### Confidence output  

The model outputs confidence in the range `[0, 1]` and represent the ARel error after affine matching with GT. The confidence itself is shift invariance, namely the confidence is a ranking and relative within one input. In particular, it does not have an absolute meaning (e.g. no heteroschdastic noise modelling).


### Decoder design predicting separately scale-shift invariant depth and scale and shift to allow more diverse training. 

The decoder presents three heads: `Camera`, `Depth` and `Global`. `Depth` head predicts scale and shift invariant depth: exponential of normalized values. `Global` head predicts the scale and shift to matche the `Depth` head output to metric. With such design we can mix seamlessly dataset with metric GT, scale-invariant (i.e., SfM) or scale-shift invariant by turning down the gradient to the `Global` head when GT is either scale or shift invariant. This allows to scale up the training variety. On the other hand, more diversity is linked to lower GT quality, thus introducing artifacts...


### Faster inference  

The model is >30% faster than V1, tested on RTX4090 with float16 data-type.
