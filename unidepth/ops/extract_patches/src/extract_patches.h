#pragma once

#include "cpu/extract_patches_cpu.h"

#ifdef WITH_CUDA
#include "cuda/extract_patches_cuda.h"
#endif

#include <vector>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor extract_patches_forward(
    const torch::Tensor &images,
    const torch::Tensor &coords,
    int patch_height,
    int patch_width)
{
    if (images.type().is_cuda())
    {
#ifdef WITH_CUDA
        return extract_patches_cuda_forward(images, coords, patch_height, patch_width);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> extract_patches_backward(
    const torch::Tensor &grad_patches,
    const torch::Tensor &coords,
    int H,
    int W)
{
    if (grad_patches.type().is_cuda())
    {
#ifdef WITH_CUDA
        return extract_patches_cuda_backward(grad_patches, coords, H, W);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}