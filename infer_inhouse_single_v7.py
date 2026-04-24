"""
infer_inhouse_single_v7.py
==========================
Single-pair inference for SkinAnalyzerV2 checkpoints trained by skin_train_v7.py.

Differences from infer_inhouse_v7.py
------------------------------------
- accepts exactly one cross/parallel image pair
- does not accept GT inputs
- saves scores / optional visualization / optional binary masks
- always saves illumination diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from infer_inhouse_v7 import (
    extract_illumination_diagnostics,
    load_model_v7,
    save_illumination_diagnostics,
)
from skin_infer import infer_single, load_rgb_full, save_visualization
from skin_mask_gen import generate_skin_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Single in-house inference for skin_train_v7 checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--cross', required=True, help='cross polarization image path')
    parser.add_argument('--parallel', required=True, help='parallel polarization image path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint path (.pth)')
    parser.add_argument('--out_dir', required=True, help='output directory')

    parser.add_argument('--stem', default=None, help='output stem name (default: cross filename stem)')
    parser.add_argument('--img_size', type=int, default=256, help='single-pass inference size for small images')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size for large-image inference')
    parser.add_argument('--overlap', type=int, default=64, help='patch overlap')
    parser.add_argument('--patch_batch_size', type=int, default=8, help='number of patches per inference batch')
    parser.add_argument('--mask_threshold', type=float, default=0.5, help='threshold for saving binary masks')

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use')
    parser.add_argument('--save_vis', action='store_true', help='save visualization panel')
    parser.add_argument('--save_masks', action='store_true', help='save binary prediction masks')
    parser.add_argument('--no_mediapipe', action='store_true', help='disable MediaPipe skin-mask generation fallback')
    parser.add_argument(
        '--face_landmarker_task_path',
        default=None,
        help='MediaPipe face_landmarker.task path; when set, use new_data_prep-compatible mask generation',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cross_path = Path(args.cross)
    parallel_path = Path(args.parallel)
    if not cross_path.exists():
        raise FileNotFoundError(f'cross image not found: {cross_path}')
    if not parallel_path.exists():
        raise FileNotFoundError(f'parallel image not found: {parallel_path}')

    stem = args.stem or cross_path.stem
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Device: {device}')
    print('Loading v7-compatible model checkpoint...')
    model, cfg = load_model_v7(args.checkpoint, device)

    img_size = int(cfg.get('img_size', args.img_size))
    patch_size = int(args.patch_size)
    overlap = int(args.overlap)
    patch_batch_size = int(args.patch_batch_size)

    vis_dir = out_dir / 'visualizations'
    mask_dir = out_dir / 'masks'
    illum_dir = out_dir / 'illumination_diagnostics'

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        for tag in ('skin', 'brown', 'red', 'wrinkle'):
            (mask_dir / tag).mkdir(parents=True, exist_ok=True)
    illum_dir.mkdir(parents=True, exist_ok=True)

    print(f'Infer: {stem}')
    rgb_cross_full = load_rgb_full(cross_path)
    skin_mask = generate_skin_mask(
        rgb_cross_full,
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
        TF.to_pil_image((wrinkle_prob >= args.mask_threshold).float()).save(mask_dir / 'wrinkle' / f'{stem}.png')

    rgb_parallel_full = load_rgb_full(parallel_path)
    diag = extract_illumination_diagnostics(
        model=model,
        rgb_cross_full=rgb_cross_full,
        rgb_parallel_full=rgb_parallel_full,
        device=device,
    )
    save_illumination_diagnostics(diag, illum_dir / f'{stem}_illum.png')

    score_item = {
        'stem': stem,
        'cross_path': str(cross_path),
        'parallel_path': str(parallel_path),
        'brown_score': round(pred['brown_score'], 4),
        'red_score': round(pred['red_score'], 4),
        'wrinkle_score': round(pred['wrinkle_score'], 4),
    }

    print('Saving reports...')
    with (out_dir / 'scores.json').open('w', encoding='utf-8') as f:
        json.dump([score_item], f, indent=2, ensure_ascii=False)

    with (out_dir / 'result.json').open('w', encoding='utf-8') as f:
        json.dump(score_item, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == '__main__':
    main()
