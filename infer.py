import argparse
import os

import cv2
import numpy as np
import torch
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image

from cvcore.config import get_cfg_defaults
from cvcore.model import build_model


RIB_LABELS = [
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10",
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run rib segmentation on a CXR image.")
    parser.add_argument("--config", required=True, help="Path to config yaml.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--image", required=True, help="Path to input CXR image.")
    parser.add_argument("--output", default="outputs/infer_overlay.png",
                        help="Path to save overlay visualization.")
    parser.add_argument("--mask-output", default="",
                        help="Optional path to save merged binary mask.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for segmentation masks.")
    parser.add_argument("--device", default="cuda",
                        help="Inference device, for example cuda or cpu.")
    return parser.parse_args()


def load_state_dict(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    stripped_state_dict = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(stripped_state_dict)


def load_image(image_path):
    image = Image.open(image_path).convert("L")
    image_np = np.asarray(image, dtype=np.float32)
    model_input = np.expand_dims(image_np / 255.0, axis=2)
    return image_np, model_input


def predict_masks(model, image, threshold, device):
    transform = Compose([Resize(512, 512), ToTensorV2()])
    tensor = transform(image=image)["image"].unsqueeze(0).to(device=device, dtype=torch.float)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)
        masks = (probabilities > threshold).squeeze(0).cpu().numpy().astype(np.uint8)

    return masks


def make_overlay(image_np, masks):
    height, width = image_np.shape[:2]
    resized_masks = np.stack([
        cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        for mask in masks
    ])

    overlay = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    colors = cv2.applyColorMap(
        np.linspace(0, 255, len(RIB_LABELS), dtype=np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO,
    )[:, 0, ::-1]

    alpha = 0.45
    for mask, color in zip(resized_masks, colors):
        area = mask > 0
        overlay[area] = ((1 - alpha) * overlay[area] + alpha * color).astype(np.uint8)

    merged_mask = resized_masks.any(axis=0).astype(np.uint8) * 255
    return overlay, merged_mask


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    model = build_model(cfg).to(device)
    load_state_dict(model, args.checkpoint, device)
    model.eval()

    image_np, model_input = load_image(args.image)
    masks = predict_masks(model, model_input, args.threshold, device)
    overlay, merged_mask = make_overlay(image_np, masks)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(overlay).save(args.output)

    if args.mask_output:
        os.makedirs(os.path.dirname(args.mask_output) or ".", exist_ok=True)
        Image.fromarray(merged_mask).save(args.mask_output)

    print(f"Saved overlay to {args.output}")
    if args.mask_output:
        print(f"Saved merged mask to {args.mask_output}")


if __name__ == "__main__":
    main()
