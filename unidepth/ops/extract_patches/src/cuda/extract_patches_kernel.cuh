#ifndef EXTRACT_PATCHES_KERNEL_CUH
#define EXTRACT_PATCHES_KERNEL_CUH

#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // should contain __half

// Declare the forward CUDA kernel function
template <typename T> __global__ void extract_patches_cuda_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int* __restrict__ centers,
    int B, int C, int H, int W,
    int N, int h, int w);

// Declare the backward CUDA kernel function
template <typename T> __global__ void extract_patches_cuda_backward_kernel(
    const T* __restrict__ grad_output,
    T* __restrict__ grad_input,
    const int* __restrict__ centers,
    int B, int C, int H, int W,
    int N, int h, int w);

#endif // EXTRACT_PATCHES_KERNEL_CUH
