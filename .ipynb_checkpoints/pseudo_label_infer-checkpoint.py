import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import backbones
from glass import GLASS


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def get_polar_center(h, w, center_x=-1.0, center_y=-1.0):
    cx = (w - 1) / 2.0 if center_x < 0 else center_x
    cy = (h - 1) / 2.0 if center_y < 0 else center_y
    cx = float(np.clip(cx, 0.0, w - 1))
    cy = float(np.clip(cy, 0.0, h - 1))
    return cx, cy


def warp_polar_image(img, center_x=-1.0, center_y=-1.0, max_radius_ratio=1.0, inverse=False, interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    cx, cy = get_polar_center(h, w, center_x, center_y)
    max_radius = min(cx, cy, (w - 1) - cx, (h - 1) - cy)
    max_radius = max(1.0, max_radius * max_radius_ratio)

    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS + interpolation
    if inverse:
        flags += cv2.WARP_INVERSE_MAP

    return cv2.warpPolar(img, (w, h), (cx, cy), max_radius, flags)


def build_transform(imagesize):
    return transforms.Compose(
        [
            transforms.Resize(imagesize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def collect_images(test_dir):
    exts = ("*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(test_dir, "**", ext), recursive=True))
    return sorted(paths)


def load_model(args, device):
    backbone = backbones.load(args.backbone)
    model = GLASS(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=args.layers,
        device=device,
        input_shape=(3, args.imagesize, args.imagesize),
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        patchsize=args.patchsize,
        patchstride=1,
        meta_epochs=1,
        eval_epochs=1,
        dsc_layers=args.dsc_layers,
        dsc_hidden=args.dsc_hidden,
        dsc_margin=0.5,
        train_backbone=False,
        pre_proj=args.pre_proj,
        mining=0,
        noise=0.0,
        radius=0.75,
        p=0.0,
        lr=1e-4,
        svd=0,
        step=1,
        limit=1,
    )

    state_dict = torch.load(args.ckpt_path, map_location=device)
    if "pre_projection" in state_dict and hasattr(model, "pre_projection"):
        model.pre_projection.load_state_dict(state_dict["pre_projection"], strict=False)
    if "discriminator" in state_dict:
        model.discriminator.load_state_dict(state_dict["discriminator"], strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    if hasattr(model, "forward_modules"):
        model.forward_modules.eval()
    if hasattr(model, "pre_projection"):
        model.pre_projection.eval()
    model.discriminator.eval()

    return model


def infer_anomaly_map(model, img_tensor, device):
    with torch.no_grad():
        batch = img_tensor.unsqueeze(0).to(torch.float).to(device)
        feat_map, _ = model._embed(batch, provide_patch_shapes=True, evaluation=True)
        if getattr(model, "pre_proj", 0) > 0 and hasattr(model, "pre_projection"):
            feat_map = model.pre_projection(feat_map)

        score_map = model.discriminator(feat_map).squeeze(1)
        mask = model.anomaly_segmentor.convert_to_segmentation(score_map)[0]
        image_score = float(score_map.reshape(score_map.shape[0], -1).max(dim=1).values[0].cpu().item())
    return mask.astype(np.float32), image_score


def save_binary_mask(mask, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)


def make_vis(raw_bgr, heatmap_float, binary_u8):
    heat = np.clip(heatmap_float, 0.0, 1.0)
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    binary_color = cv2.cvtColor(binary_u8, cv2.COLOR_GRAY2BGR)

    vis = np.hstack([raw_bgr, heat_color, binary_color])
    return vis


def main():
    parser = argparse.ArgumentParser("Pseudo mask inference for GLASS")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_mask_dir", type=str, required=True)
    parser.add_argument("--output_vis_dir", type=str, default="")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--imagesize", type=int, default=512)
    parser.add_argument("--resize", type=int, default=512)

    parser.add_argument("--backbone", type=str, default="wideresnet50")
    parser.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1536)
    parser.add_argument("--target_embed_dimension", type=int, default=1536)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--dsc_layers", type=int, default=2)
    parser.add_argument("--dsc_hidden", type=int, default=1024)
    parser.add_argument("--pre_proj", type=int, default=1)

    parser.add_argument("--use_polar", type=str2bool, default=False)
    parser.add_argument("--polar_center_x", type=float, default=-1.0)
    parser.add_argument("--polar_center_y", type=float, default=-1.0)
    parser.add_argument("--polar_max_radius_ratio", type=float, default=1.0)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_mode", type=str, default="fixed", choices=["fixed", "percentile"])
    parser.add_argument("--threshold_percentile", type=float, default=99.5)
    parser.add_argument("--min_area", type=int, default=0)

    parser.add_argument("--save_vis_compare", type=str2bool, default=True)
    parser.add_argument("--vis_save_size", type=int, default=0, help="0 keeps original size; >0 resizes visualization height to this value")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    test_dir = Path(args.test_dir)
    output_mask_dir = Path(args.output_mask_dir)
    output_vis_dir = Path(args.output_vis_dir) if args.output_vis_dir else None

    image_paths = collect_images(str(test_dir))
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under: {test_dir}")

    model = load_model(args, device)
    transform = build_transform(args.resize)

    raw_bgr_list = []
    rel_paths = []
    heatmaps_cart = []
    scores = []

    for p in image_paths:
        raw_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if raw_bgr is None:
            continue

        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        model_in_rgb = raw_rgb
        if args.use_polar:
            model_in_rgb = warp_polar_image(
                model_in_rgb,
                center_x=args.polar_center_x,
                center_y=args.polar_center_y,
                max_radius_ratio=args.polar_max_radius_ratio,
                inverse=False,
                interpolation=cv2.INTER_LINEAR,
            )

        pil_img = Image.fromarray(model_in_rgb)
        x = transform(pil_img)

        heatmap_model, image_score = infer_anomaly_map(model, x, device)

        # heatmap_model is in model input domain (possibly polar); bring to original Cartesian domain
        heatmap_resized = cv2.resize(heatmap_model, (raw_rgb.shape[1], raw_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        if args.use_polar:
            heatmap_cart = warp_polar_image(
                heatmap_resized,
                center_x=args.polar_center_x,
                center_y=args.polar_center_y,
                max_radius_ratio=args.polar_max_radius_ratio,
                inverse=True,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            heatmap_cart = heatmap_resized

        raw_bgr_list.append(raw_bgr)
        rel_paths.append(Path(p).relative_to(test_dir))
        heatmaps_cart.append(np.clip(heatmap_cart, 0.0, 1.0))
        scores.append(image_score)

    if len(heatmaps_cart) == 0:
        raise RuntimeError("No valid images were processed.")

    # determine threshold
    if args.threshold_mode == "fixed":
        thr = args.threshold
    else:
        all_vals = np.concatenate([h.reshape(-1) for h in heatmaps_cart], axis=0)
        thr = float(np.percentile(all_vals, args.threshold_percentile))

    stats_lines = ["image_path,image_score,threshold,mask_area_ratio"]

    for raw_bgr, rel_path, heat, image_score in zip(raw_bgr_list, rel_paths, heatmaps_cart, scores):
        binary = (heat >= thr).astype(np.uint8) * 255

        if args.min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
            refined = np.zeros_like(binary, dtype=np.uint8)
            for idx in range(1, num_labels):
                area = stats[idx, cv2.CC_STAT_AREA]
                if area >= args.min_area:
                    refined[labels == idx] = 255
            binary = refined

        save_mask_path = output_mask_dir / rel_path
        save_mask_path = save_mask_path.with_suffix(".png")
        save_binary_mask(binary, save_mask_path)

        area_ratio = float((binary > 0).mean())
        stats_lines.append(f"{rel_path.as_posix()},{image_score:.6f},{thr:.6f},{area_ratio:.6f}")

        if args.save_vis_compare and output_vis_dir is not None:
            vis = make_vis(raw_bgr, heat, binary)
            if args.vis_save_size > 0:
                vis = cv2.resize(vis, (args.vis_save_size * 3, args.vis_save_size), interpolation=cv2.INTER_LINEAR)
            save_vis_path = output_vis_dir / rel_path
            save_vis_path = save_vis_path.with_suffix(".png")
            save_vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_vis_path), vis)

    output_mask_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_mask_dir / "pseudo_stats.csv"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))

    print(f"Done. Saved pseudo masks to: {output_mask_dir}")
    if args.save_vis_compare and output_vis_dir is not None:
        print(f"Saved visual comparisons to: {output_vis_dir}")
    print(f"Stats CSV: {stats_path}")
    print(f"Threshold used: {thr:.6f}")


if __name__ == "__main__":
    main()
