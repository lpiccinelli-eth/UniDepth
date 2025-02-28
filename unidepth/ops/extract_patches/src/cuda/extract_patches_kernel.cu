#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cuda/extract_patches_kernel.cuh"
#include "cuda/extract_patches_cuda.h"


// Need to templetize these two to get fp16 working, but problems in compilation...
torch::Tensor extract_patches_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &centers,
    int h,
    int w
) {
    
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int N = centers.size(1);

    auto output = torch::zeros({B, C, N, h, w}, input.options());

    const int threads = C;
    const dim3 blocks(B, N);

    extract_patches_cuda_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        centers.data_ptr<int>(),
        B, C, H, W,
        N, h, w);

    return {output};
}

std::vector<torch::Tensor> extract_patches_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &centers,
    int H,
    int W
) {
    
    int B = grad_output.size(0);
    int C = grad_output.size(1);
    int N = centers.size(1);
    int h = grad_output.size(3);
    int w = grad_output.size(4);

    auto grad_input = torch::zeros({B, C, H, W}, grad_output.options());

    const int threads = C;
    const dim3 blocks(B, N);

    extract_patches_cuda_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        centers.data_ptr<int>(),
        B, C, H, W,
        N, h, w);

    return {grad_input};
}

template <typename T>
__global__ void extract_patches_cuda_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int* __restrict__ centers,
    int B, int C, int H, int W,
    int N, int h, int w) {
    
    // Calculate thread indices
    int batch_idx = blockIdx.x;
    int patch_idx = blockIdx.y;
    int channel_idx = threadIdx.x;

    // Extract center coordinates
    int center_y = centers[(batch_idx * N + patch_idx) * 2];
    int center_x = centers[(batch_idx * N + patch_idx) * 2 + 1];

    // Calculate half patch size
    int half_h = h / 2;
    int half_w = w / 2;

    // Extract patch
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            int y = center_y - half_h + i;
            int x = center_x - half_w + j;
            output[batch_idx * C * N * h * w + patch_idx * C * h * w + channel_idx * h * w + i * w + j] = 
                input[batch_idx * C * H * W + channel_idx * H * W + y * W + x];
        }
    }
}

template __global__ void extract_patches_cuda_forward_kernel<float>(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ centers,
    int B, int C, int H, int W,
    int N, int h, int w);

template __global__ void extract_patches_cuda_forward_kernel<__half>(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const int* __restrict__ centers,
    int B, int C, int H, int W,
    int N, int h, int w);

template <typename T>
__global__ void extract_patches_cuda_backward_kernel(
    const T* __restrict__ grad_output,
    T* __restrict__ grad_input,
    const int* __restrict__ centers,
    int B, int C, int H, int W,
    int N, int h, int w) {
    
    // Calculate thread indices
    int batch_idx = blockIdx.x;
    int patch_idx = blockIdx.y;
    int channel_idx = threadIdx.x;

    // Extract center coordinates
    int center_y = centers[(batch_idx * N + patch_idx) * 2];
    int center_x = centers[(batch_idx * N + patch_idx) * 2 + 1];

    // Calculate half patch size
    int half_h = h / 2;
    int half_w = w / 2;

    // Compute gradients with respect to input tensor using chain rule
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            int y = center_y - half_h + i;
            int x = center_x - half_w + j;
            
            atomicAdd(
                &grad_input[batch_idx * C * H * W + channel_idx * H * W + y * W + x],
                grad_output[batch_idx * C * N * h * w + patch_idx * C * h * w + channel_idx * h * w + i * w + j]
            );
        }
    }
}

template __global__ void extract_patches_cuda_backward_kernel<float>(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const int* __restrict__ centers,
    int B, int C, int H, int W,
    int N, int h, int w);

template __global__ void extract_patches_cuda_backward_kernel<__half>(
    const __half* __restrict__ grad_output,
    __half* __restrict__ grad_input,
    const int* __restrict__ centers,
    int B, int C, int H, int W,
    int N, int h, int w);