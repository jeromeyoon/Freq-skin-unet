"""
infer_inhouse_smooth_v2.py
==========================
In-house dataset full inference for smooth/stable checkpoints
trained by skin_train_smooth_v2.py.

This follows the infer_inhouse.py workflow, but loads the smooth/stable
SkinAnalyzer architecture that depends on ir_encoder_stable injection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import ir_encoder_stable as _ir_stable

sys.modules['ir_encoder'] = _ir_stable

from in_house_dataload import build_file_list
from infer_inhouse import (
    _print_summary,
    _save_dice_summary,
    _save_scores_json,
    run_inhouse_infer,
)
from skin_net import build_analyzer


def load_model_smooth_v2(checkpoint_path: str, device: torch.device):
    """Load a SkinAnalyzer checkpoint trained by skin_train_smooth/smooth_v2."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('cfg', {})

    model = build_analyzer(
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
        description='In-house full inference for skin_train_smooth_v2 checkpoints',
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

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use')
    parser.add_argument('--save_vis', action='store_true',
                        help='save visualization panels')
    parser.add_argument('--save_masks', action='store_true',
                        help='save binary prediction masks')
    parser.add_argument('--no_mediapipe', action='store_true',
                        help='disable MediaPipe skin-mask generation fallback')
    return parser.parse_args()


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

    print("Loading smooth_v2-compatible model checkpoint...")
    model, cfg = load_model_smooth_v2(args.checkpoint, device)

    img_size = int(cfg.get('img_size', args.img_size))
    patch_size = int(args.patch_size)
    overlap = int(args.overlap)
    patch_batch_size = int(args.patch_batch_size)

    print("Running inference...")
    all_scores = run_inhouse_infer(
        records=records,
        model=model,
        out_dir=out_dir,
        device=device,
        img_size=img_size,
        patch_size=patch_size,
        overlap=overlap,
        patch_batch_size=patch_batch_size,
        save_vis=args.save_vis,
        save_masks=args.save_masks,
        mask_threshold=args.mask_threshold,
        use_mediapipe=not args.no_mediapipe,
    )

    print("Saving reports...")
    _save_scores_json(all_scores, out_dir / 'scores.json')
    _save_dice_summary(all_scores, out_dir / 'dice_scores.txt')
    _print_summary(all_scores)

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == '__main__':
    main()
