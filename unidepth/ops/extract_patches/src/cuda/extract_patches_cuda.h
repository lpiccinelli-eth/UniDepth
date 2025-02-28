#ifndef EXTRACT_PATCHES_CUDA_H
#define EXTRACT_PATCHES_CUDA_H

#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Function prototypes for the CUDA functions
torch::Tensor extract_patches_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &centers,
    int h,
    int w
);

std::vector<torch::Tensor> extract_patches_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &centers,
    int H,
    int W
);


#endif // EXTRACT_PATCHES_CUDA_H
