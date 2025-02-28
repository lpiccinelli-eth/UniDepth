import RandomPatchExtraction
import torch
from torch.autograd import Function


class ExtractPatchesFunction(Function):
    @staticmethod
    def forward(ctx, input, centers, h, w):
        # Save variables for backward pass. inputs for shapes
        ctx.save_for_backward(input, centers)

        return RandomPatchExtraction.extract_patches_forward(input, centers, h, w)

    @staticmethod
    def backward(ctx, grad_output):
        input, centers = ctx.saved_tensors

        (grad_input,) = RandomPatchExtraction.extract_patches_backward(
            grad_output, centers, input.shape[2], input.shape[3]
        )
        # breakpoint()

        # Return gradients with respect to inputs only
        return grad_input, None, None, None


# Test
if __name__ == "__main__":
    B, C, H, W = 1, 1, 10, 10
    N = 2
    h, w = 3, 3
    input = torch.arange(
        B * C * H * W, device="cuda", dtype=torch.float32, requires_grad=True
    ).view(B, C, H, W)
    centers = torch.tensor([[[4, 4], [6, 6]]], device="cuda", dtype=torch.int32)
    output = ExtractPatchesFunction.apply(input, centers, h, w)
    output.mean().backward()
