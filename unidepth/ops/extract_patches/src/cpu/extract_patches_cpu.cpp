#include <vector>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor extract_patches_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &centers,
    int h,
    int w
) {
    AT_ERROR("Not implement on cpu");
}

std::vector<at::Tensor> extract_patches_cpu_backward(
    const torch::Tensor &grad_patches,
    const torch::Tensor &coords,
    int H,
    int W
) {
    AT_ERROR("Not implement on cpu");
}
