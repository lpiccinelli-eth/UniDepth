import torch
import numpy as np
from PIL import Image

from unidepth.utils import colorize, image_grid


def demo(model):
    rgb = np.array(Image.open("assets/demo/rgb.png"))
    import cv2
    rgb = cv2.resize(rgb, (616, 462))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))

    # predict
    # predictions = model.infer(rgb_torch, intrinsics_torch)
    with torch.no_grad():
        predictions = model({"image": rgb_torch.unsqueeze(0).float().cuda(), "K": intrinsics_torch.unsqueeze(0).cuda()}, {})
    print(predictions.keys())
    # get GT and pred
    depth_pred = predictions["depth"].squeeze().cpu().numpy()
    depth_gt = np.array(Image.open("assets/demo/depth.png")).astype(float) / 1000.0

    # compute error, you have zero divison where depth_gt == 0.0
    depth_arel = np.abs(depth_gt - depth_pred) / depth_gt
    depth_arel[depth_gt == 0.0] = 0.0

    # colorize
    depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")
    depth_gt_col = colorize(depth_gt, vmin=0.01, vmax=10.0, cmap="magma_r")
    depth_error_col = colorize(depth_arel, vmin=0.0, vmax=0.2, cmap="coolwarm")

    # save image with pred and error
    artifact = image_grid([rgb, depth_gt_col, depth_pred_col, depth_error_col], 2, 2)
    Image.fromarray(artifact).save("assets/demo/output.png")

    print("Available predictions:", list(predictions.keys()))
    print(f"ARel: {depth_arel[depth_gt > 0].mean() * 100:.2f}%")


if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    model = torch.hub.load(
        "lpiccinelli-eth/unidepth:feat/fwd",
        "UniDepthV1_ViTL14",
        pretrained=True,
        trust_repo=True,
        force_reload=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    demo(model)
