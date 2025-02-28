import RandomPatchExtraction
import torch


def extract_patches(input, centers, patch_size):
    h, w = patch_size
    output = RandomPatchExtraction.extract_patches_forward(input, centers, h, w)
    breakpoint()
    return output


# Example usage
if __name__ == "__main__":
    B, C, H, W = 1, 1, 10, 10
    N = 2
    h, w = 3, 3
    input = torch.arange(
        B * C * H * W, device="cuda", dtype=torch.float32, requires_grad=True
    ).view(B, C, H, W)
    centers = torch.tensor([[[4, 4], [6, 6]]], device="cuda", dtype=torch.int32)
    patches = extract_patches(input, centers, (h, w))
