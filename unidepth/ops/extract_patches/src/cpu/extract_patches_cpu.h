#pragma once
#include <torch/extension.h>
#include <vector>


torch::Tensor extract_patches_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &centers,
    int h,
    int w
);

std::vector<at::Tensor> extract_patches_cpu_backward(
    const torch::Tensor &grad_patches,
    const torch::Tensor &coords,
    int H,
    int W
);
