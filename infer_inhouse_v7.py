"""
infer_inhouse_v7.py
===================
In-house dataset full inference for SkinAnalyzerV2 checkpoints
trained by skin_train_v7.py.

This script follows the infer_inhouse.py workflow, but loads the
V2/V7 model architecture from skin_net_v2.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.utils as vutils

from in_house_dataload import build_file_list
from infer_inhouse import (
    _load_gt_aligned,
    _print_summary,
    _save_dice_summary,
    _save_scores_json,
)
from skin_infer import compute_dice, infer_single, load_rgb_full, save_visualization
from skin_mask_gen import generate_skin_mask
from skin_net_v2 import build_analyzer_v2


def load_model_v7(checkpoint_path: str, device: torch.device):
    """Load a SkinAnalyzerV2 checkpoint trained by skin_train_v7."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('cfg', {})

    model = build_analyzer_v2(
        base_ch=cfg.get('base_ch', 64),
        low_r=cfg.get('low_r', 0.1),
        high_r=cfg.get('high_r', 0.4),
    ).to(device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    val = ckpt.get('val_loss', {})
    val_total = val.get('total', None)
    if isinstance(val_total, (int, float)):
        print(f"Checkpoint loaded  epoch={epoch}  val_loss={val_total:.4f}")
    else:
        print(f"Checkpoint loaded  epoch={epoch}")

    return model, cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='In-house full inference for skin_train_v7 checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--input_root', required=True,
                        help='input image root (D*/subject*/F_10.jpg structure)')
    parser.add_argument('--checkpoint', required=True,
                        help='checkpoint path (.pth)')
    parser.add_argument('--out_dir', required=True,
                        help='output directory')

    parser.add_argument('--brown_root', default=None,
                        help='brown GT mask directory')
    parser.add_argument('--red_root', default=None,
                        help='red GT mask directory')
    parser.add_argument('--wrinkle_root', default=None,
                        help='wrinkle GT mask directory')

    parser.add_argument('--cross_name', default='F_10.jpg',
                        help='cross polarization image filename')
    parser.add_argument('--parallel_name', default='F_11.jpg',
                        help='parallel polarization image filename')

    parser.add_argument('--img_size', type=int, default=256,
                        help='single-pass inference size for small images')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='patch size for large-image inference')
    parser.add_argument('--overlap', type=int, default=64,
                        help='patch overlap')
    parser.add_argument('--patch_batch_size', type=int, default=8,
                        help='number of patches per inference batch')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                        help='threshold for saving binary masks')
    parser.add_argument('--wrinkle_threshold', type=float, default=0.4,
                        help='lower threshold for wrinkle (use with --wrinkle_thin)')
    parser.add_argument('--wrinkle_thin', action='store_true',
                        help='apply morphological thinning to wrinkle mask for line output')

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use')
    parser.add_argument('--save_vis', action='store_true',
                        help='save visualization panels')
    parser.add_argument('--save_masks', action='store_true',
                        help='save binary prediction masks')
    parser.add_argument('--no_mediapipe', action='store_true',
                        help='disable MediaPipe skin-mask generation fallback')
    parser.add_argument('--face_landmarker_task_path', default=None,
                        help='MediaPipe face_landmarker.task path; when set, use new_data_prep-compatible mask generation')
    return parser.parse_args()


def _normalize_vis(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor for visualization.
    Accepts [1,H,W] or [3,H,W] and returns [3,H,W].
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor [C,H,W], got shape={tuple(x.shape)}")

    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    elif x.shape[0] > 3:
        x = x[:3]

    x = x.float()
    x_min = x.amin(dim=(1, 2), keepdim=True)
    x_max = x.amax(dim=(1, 2), keepdim=True)
    return ((x - x_min) / (x_max - x_min + 1e-6)).clamp(0.0, 1.0)


@torch.no_grad()
def extract_illumination_diagnostics(
    model,
    rgb_cross_full: torch.Tensor,
    rgb_parallel_full: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Extract intermediate representations useful for checking whether
    ambient illumination has been suppressed.
    """
    cross = rgb_cross_full.unsqueeze(0).to(device)
    parallel = rgb_parallel_full.unsqueeze(0).to(device)

    eps = getattr(model.cross_encoder, 'eps', 1e-3)
    od = -torch.log(cross.clamp(min=eps, max=1.0)).clamp(-6.0, 6.0)
    od_chroma = model.cross_encoder.od_filter(od)
    parallel_pre = model.parallel_encoder.preprocess(parallel)

    return {
        'cross_rgb': rgb_cross_full.cpu(),
        'cross_od_chroma': od_chroma[0].cpu(),
        'parallel_rgb': rgb_parallel_full.cpu(),
        'parallel_preprocess': parallel_pre[0].cpu(),
    }


def save_illumination_diagnostics(diag: dict, save_path: Path, max_vis_size: int = 1024) -> None:
    items = [
        _normalize_vis(diag['cross_rgb']),
        _normalize_vis(diag['cross_od_chroma']),
        _normalize_vis(diag['parallel_rgb']),
        _normalize_vis(diag['parallel_preprocess']),
    ]

    resized = []
    for x in items:
        h = x.shape[1]
        if h > max_vis_size:
            scale = max_vis_size / h
            new_h = max_vis_size
            new_w = max(1, int(x.shape[2] * scale))
            x = F.interpolate(x.unsqueeze(0), (new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        resized.append(x)

    grid = vutils.make_grid(torch.stack(resized), nrow=4, padding=2, normalize=False)
    TF.to_pil_image(grid).save(save_path)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Building input file list...")
    records = build_file_list(
        input_root=args.input_root,
        brown_root=args.brown_root,
        red_root=args.red_root,
        wrinkle_root=args.wrinkle_root,
        cross_name=args.cross_name,
        parallel_name=args.parallel_name,
    )
    if not records:
        raise RuntimeError("No valid in-house records found.")

    print("Loading v7-compatible model checkpoint...")
    model, cfg = load_model_v7(args.checkpoint, device)

    img_size = int(cfg.get('img_size', args.img_size))
    patch_size = int(args.patch_size)
    overlap = int(args.overlap)
    patch_batch_size = int(args.patch_batch_size)

    print("Running inference...")

    vis_dir = out_dir / 'visualizations'
    mask_dir = out_dir / 'masks'
    illum_dir = out_dir / 'illumination_diagnostics'

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        for tag in ('skin', 'brown', 'red', 'wrinkle'):
            (mask_dir / tag).mkdir(parents=True, exist_ok=True)
    illum_dir.mkdir(parents=True, exist_ok=True)

    all_scores = []

    for rec in records:
        stem = rec['stem']
        cross_path = rec['rgb_cross']
        parallel_path = rec['rgb_parallel']

        print(f"Infer: {stem}")

        rgb_full = load_rgb_full(cross_path)
        skin_mask = generate_skin_mask(
            rgb_full,
            use_mediapipe=not args.no_mediapipe,
            face_landmarker_task_path=args.face_landmarker_task_path,
        )

        pred = infer_single(
            model=model,
            cross_path=cross_path,
            parallel_path=parallel_path,
            img_size=img_size,
            device=device,
            skin_mask=skin_mask,
            patch_size=patch_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
        )

        brown_prob = pred['brown_mask']
        red_prob = pred['red_mask']
        wrinkle_prob = pred['wrinkle_mask']
        h, w = brown_prob.shape[1], brown_prob.shape[2]

        dice: dict[str, float | None] = {}
        pred_key_map = {'brown': brown_prob, 'red': red_prob, 'wrinkle': wrinkle_prob}
        for task in ('brown', 'red', 'wrinkle'):
            gt_path = rec[task]
            if gt_path is not None:
                gt_tensor = _load_gt_aligned(gt_path, h, w)
                dice[task] = compute_dice(pred_key_map[task], gt_tensor, skin_mask=skin_mask)
            else:
                dice[task] = None

        if args.save_vis:
            save_visualization(
                rgb_cross=pred['rgb_cross'],
                brown_prob=brown_prob,
                red_prob=red_prob,
                wrinkle_prob=wrinkle_prob,
                save_path=vis_dir / f'{stem}_vis.png',
                brown_score=pred['brown_score'],
                red_score=pred['red_score'],
                wrinkle_score=pred['wrinkle_score'],
            )

        if args.save_masks:
            if skin_mask.shape[1] != h or skin_mask.shape[2] != w:
                skin_save = F.interpolate(skin_mask.unsqueeze(0), (h, w), mode='nearest').squeeze(0)
            else:
                skin_save = skin_mask
            TF.to_pil_image(skin_save).save(mask_dir / 'skin' / f'{stem}.png')
            TF.to_pil_image((brown_prob >= args.mask_threshold).float()).save(mask_dir / 'brown' / f'{stem}.png')
            TF.to_pil_image((red_prob >= args.mask_threshold).float()).save(mask_dir / 'red' / f'{stem}.png')
            # Wrinkle: optionally thin the binary mask to line representation.
            # Lower threshold captures more wrinkles before thinning discards width.
            wrinkle_thr = args.wrinkle_threshold if args.wrinkle_thin else args.mask_threshold
            wrinkle_bin = (wrinkle_prob >= wrinkle_thr).float()
            if args.wrinkle_thin:
                try:
                    import cv2
                    import numpy as np
                    arr = (wrinkle_bin.squeeze().numpy() * 255).astype(np.uint8)
                    thinned = cv2.ximgproc.thinning(arr)
                    wrinkle_bin = torch.from_numpy(thinned / 255.0).unsqueeze(0).float()
                except Exception:
                    pass  # fallback: save without thinning if opencv-contrib unavailable
            TF.to_pil_image(wrinkle_bin).save(mask_dir / 'wrinkle' / f'{stem}.png')

        rgb_parallel_full = load_rgb_full(parallel_path)
        diag = extract_illumination_diagnostics(
            model=model,
            rgb_cross_full=rgb_full,
            rgb_parallel_full=rgb_parallel_full,
            device=device,
        )
        save_illumination_diagnostics(
            diag,
            illum_dir / f'{stem}_illum.png',
        )

        all_scores.append({
            'stem': stem,
            'brown_score': round(pred['brown_score'], 4),
            'red_score': round(pred['red_score'], 4),
            'wrinkle_score': round(pred['wrinkle_score'], 4),
            'brown_dice': round(dice['brown'], 4) if dice['brown'] is not None else None,
            'red_dice': round(dice['red'], 4) if dice['red'] is not None else None,
            'wrinkle_dice': round(dice['wrinkle'], 4) if dice['wrinkle'] is not None else None,
        })

    print("Saving reports...")
    _save_scores_json(all_scores, out_dir / 'scores.json')
    _save_dice_summary(all_scores, out_dir / 'dice_scores.txt')
    _print_summary(all_scores)

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == '__main__':
    main()
