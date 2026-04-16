"""
infer_inhouse.py
================
사내 데이터셋 전체 일괄 추론 + Dice Score 계산

in_house_dataload.py 로 파일 리스트를 구성한 뒤
skin_infer.py 의 패치 추론 파이프라인을 적용.
GT가 있는 샘플은 Dice score를 계산해 요약 저장.

디렉토리 구조
-------------
    input_root/
        D0531/
            007_1973_7_Male/
                F_10.jpg   (cross)
                F_11.jpg   (parallel)
    brown_root/   D05310007.png
    red_root/     D05310007.png
    wrinkle_root/ D05310007.png

출력 구조
---------
    out_dir/
        masks/
            brown/    {stem}.png
            red/      {stem}.png
            wrinkle/  {stem}.png
        visualizations/
            {stem}_vis.png
        scores.json          (전체 스코어)
        dice_scores.txt      (GT 있는 샘플 Dice 요약)

실행 예시
---------
    python infer_inhouse.py \\
        --input_root   /data/raw \\
        --brown_root   /data/gt/brown \\
        --red_root     /data/gt/red \\
        --wrinkle_root /data/gt/wrinkle \\
        --checkpoint   ./checkpoints/best_skin_analyzer.pth \\
        --out_dir      ./results/inhouse \\
        --patch_size 256 --overlap 64 --patch_batch_size 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from in_house_dataload import build_file_list
from skin_infer import (
    load_model,
    infer_single,
    compute_dice,
    save_visualization,
    load_gray_full,
)


# ══════════════════════════════════════════════════════════════════════════════
# GT 로드 (예측 마스크 해상도에 맞춰 리사이즈)
# ══════════════════════════════════════════════════════════════════════════════
def _load_gt_aligned(gt_path: Path, target_H: int, target_W: int) -> torch.Tensor:
    """
    GT 마스크를 예측 마스크 해상도에 맞춰 로드.

    - 해상도가 같으면 그대로 사용 (NEAREST 보간 없이)
    - 다르면 NEAREST 보간으로 리사이즈 (이진 마스크 경계 보호)
    """
    with Image.open(gt_path) as img:
        gW, gH = img.size

    gt = load_gray_full(gt_path)   # [1, gH, gW]

    if gH == target_H and gW == target_W:
        return gt

    # NEAREST 보간: 이진 GT 경계값 오염 방지
    gt = F.interpolate(
        gt.unsqueeze(0),
        size=(target_H, target_W),
        mode='nearest',
    ).squeeze(0)
    return gt


# ══════════════════════════════════════════════════════════════════════════════
# 결과 저장 유틸
# ══════════════════════════════════════════════════════════════════════════════
def _save_scores_json(all_scores: list[dict], path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)


def _save_dice_summary(all_scores: list[dict], path: Path) -> None:
    """GT가 있는 샘플의 Dice score 텍스트 파일 저장."""
    col_w = max((len(s['stem']) for s in all_scores), default=10) + 2

    header = (
        '# Dice Scores — 사내 데이터셋 전체 추론\n'
        '# binary Dice coefficient  (threshold = 0.5)\n'
        f"# {'stem':<{col_w}}  {'brown_dice':>12}  {'red_dice':>12}  {'wrinkle_dice':>12}\n"
        '#' + '-' * (col_w + 44)
    )

    rows = []
    for s in all_scores:
        b = f"{s['brown_dice']:.4f}"   if s['brown_dice']   is not None else 'N/A'
        r = f"{s['red_dice']:.4f}"     if s['red_dice']     is not None else 'N/A'
        w = f"{s['wrinkle_dice']:.4f}" if s['wrinkle_dice'] is not None else 'N/A'
        rows.append(f"  {s['stem']:<{col_w}}  {b:>12}  {r:>12}  {w:>12}")

    b_vals = [s['brown_dice']   for s in all_scores if s['brown_dice']   is not None]
    r_vals = [s['red_dice']     for s in all_scores if s['red_dice']     is not None]
    w_vals = [s['wrinkle_dice'] for s in all_scores if s['wrinkle_dice'] is not None]

    def _mean_line(label: str, vals: list[float]) -> str:
        if not vals:
            return f'#   {label}: N/A'
        return f'#   {label}: {sum(vals)/len(vals):.4f}  (n={len(vals)})'

    footer = (
        '#' + '-' * (col_w + 44) + '\n'
        '# 평균 (GT 있는 샘플만)\n' +
        _mean_line('mean_brown_dice  ', b_vals) + '\n' +
        _mean_line('mean_red_dice    ', r_vals) + '\n' +
        _mean_line('mean_wrinkle_dice', w_vals)
    )

    path.write_text(
        header + '\n' + '\n'.join(rows) + '\n' + footer + '\n',
        encoding='utf-8',
    )


def _print_summary(all_scores: list[dict]) -> None:
    n = len(all_scores)
    if n == 0:
        return

    avg_b = sum(s['brown_score']   for s in all_scores) / n
    avg_r = sum(s['red_score']     for s in all_scores) / n
    avg_w = sum(s['wrinkle_score'] for s in all_scores) / n
    print(f"\n{'─'*50}")
    print(f"추론 완료  총 {n}장")
    print(f"  갈색 반점 평균 스코어: {avg_b:.2f}")
    print(f"  적색 반점 평균 스코어: {avg_r:.2f}")
    print(f"  주름 평균 스코어     : {avg_w:.2f}")

    b_vals = [s['brown_dice']   for s in all_scores if s['brown_dice']   is not None]
    r_vals = [s['red_dice']     for s in all_scores if s['red_dice']     is not None]
    w_vals = [s['wrinkle_dice'] for s in all_scores if s['wrinkle_dice'] is not None]

    if any([b_vals, r_vals, w_vals]):
        print(f"\nDice 요약 (binary, threshold=0.5):")
        if b_vals:
            print(f"  mean_brown_dice   : {sum(b_vals)/len(b_vals):.4f}  (n={len(b_vals)})")
        if r_vals:
            print(f"  mean_red_dice     : {sum(r_vals)/len(r_vals):.4f}  (n={len(r_vals)})")
        if w_vals:
            print(f"  mean_wrinkle_dice : {sum(w_vals)/len(w_vals):.4f}  (n={len(w_vals)})")


# ══════════════════════════════════════════════════════════════════════════════
# 핵심 추론 루프
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inhouse_infer(
    records:          list[dict],
    model,
    out_dir:          Path,
    device:           torch.device,
    img_size:         int   = 256,
    patch_size:       int   = 256,
    overlap:          int   = 64,
    patch_batch_size: int   = 8,
    save_vis:         bool  = True,
    save_masks:       bool  = True,
    mask_threshold:   float = 0.5,
) -> list[dict]:
    """
    파일 리스트 전체에 대해 추론 + Dice 계산.

    Parameters
    ----------
    records : build_file_list() 반환값
    """
    vis_dir  = out_dir / 'visualizations'
    mask_dir = out_dir / 'masks'

    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if save_masks:
        for tag in ('brown', 'red', 'wrinkle'):
            (mask_dir / tag).mkdir(parents=True, exist_ok=True)

    all_scores: list[dict] = []
    pbar = tqdm(records, desc='사내 데이터셋 추론', unit='img', dynamic_ncols=True)

    for rec in pbar:
        stem         = rec['stem']
        cross_path   = rec['rgb_cross']
        parallel_path= rec['rgb_parallel']

        pbar.set_description(f'추론: {stem}')

        # ── 추론 ──────────────────────────────────────────────────────────────
        pred = infer_single(
            model            = model,
            cross_path       = cross_path,
            parallel_path    = parallel_path,
            mask_path        = None,          # 사내 데이터는 피부 마스크 없음 (전체 영역)
            img_size         = img_size,
            device           = device,
            patch_size       = patch_size,
            overlap          = overlap,
            patch_batch_size = patch_batch_size,
        )

        brown_prob   = pred['brown_mask']     # [1, H, W]
        red_prob     = pred['red_mask']
        wrinkle_prob = pred['wrinkle_mask']
        H, W         = brown_prob.shape[1], brown_prob.shape[2]

        # ── Dice 계산 ─────────────────────────────────────────────────────────
        dice: dict[str, float | None] = {}
        for task in ('brown', 'red', 'wrinkle'):
            gt_path = rec[task]
            if gt_path is not None:
                gt_tensor    = _load_gt_aligned(gt_path, H, W)
                pred_key_map = {'brown': brown_prob, 'red': red_prob, 'wrinkle': wrinkle_prob}
                dice[task]   = compute_dice(pred_key_map[task], gt_tensor)
            else:
                dice[task]   = None

        # ── 시각화 저장 ───────────────────────────────────────────────────────
        if save_vis:
            save_visualization(
                rgb_cross     = pred['rgb_cross'],
                brown_prob    = brown_prob,
                red_prob      = red_prob,
                wrinkle_prob  = wrinkle_prob,
                save_path     = vis_dir / f'{stem}_vis.png',
                brown_score   = pred['brown_score'],
                red_score     = pred['red_score'],
                wrinkle_score = pred['wrinkle_score'],
            )

        # ── 마스크 저장 (threshold 적용 → 이진 마스크) ────────────────────────
        if save_masks:
            brown_bin   = (brown_prob   >= mask_threshold).float()
            red_bin     = (red_prob     >= mask_threshold).float()
            wrinkle_bin = (wrinkle_prob >= mask_threshold).float()
            TF.to_pil_image(brown_bin).save(  mask_dir / 'brown'   / f'{stem}.png')
            TF.to_pil_image(red_bin).save(    mask_dir / 'red'     / f'{stem}.png')
            TF.to_pil_image(wrinkle_bin).save(mask_dir / 'wrinkle' / f'{stem}.png')

        # ── 스코어 수집 ───────────────────────────────────────────────────────
        entry = {
            'stem'         : stem,
            'brown_score'  : round(pred['brown_score'],   4),
            'red_score'    : round(pred['red_score'],     4),
            'wrinkle_score': round(pred['wrinkle_score'], 4),
            'brown_dice'   : round(dice['brown'],   4) if dice['brown']   is not None else None,
            'red_dice'     : round(dice['red'],     4) if dice['red']     is not None else None,
            'wrinkle_dice' : round(dice['wrinkle'], 4) if dice['wrinkle'] is not None else None,
        }
        all_scores.append(entry)

        pbar.set_postfix(
            b_dice=f"{dice['brown']:.3f}"   if dice['brown']   is not None else 'N/A',
            r_dice=f"{dice['red']:.3f}"     if dice['red']     is not None else 'N/A',
            w_dice=f"{dice['wrinkle']:.3f}" if dice['wrinkle'] is not None else 'N/A',
        )

    return all_scores


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='사내 데이터셋 전체 일괄 추론 + Dice Score 계산',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 필수
    parser.add_argument('--input_root',   required=True,
                        help='입력 이미지 루트 (D*/subject*/F_10.jpg 구조)')
    parser.add_argument('--checkpoint',   required=True,
                        help='학습된 체크포인트 경로 (.pth)')

    # GT 경로 (선택)
    parser.add_argument('--brown_root',   default=None, help='brownspot GT 폴더')
    parser.add_argument('--red_root',     default=None, help='red spot GT 폴더')
    parser.add_argument('--wrinkle_root', default=None, help='wrinkle GT 폴더')

    # 파일명 커스텀
    parser.add_argument('--cross_name',    default='F_10.jpg',
                        help='교차 편광 파일명 (기본: F_10.jpg)')
    parser.add_argument('--parallel_name', default='F_11.jpg',
                        help='평행 편광 파일명 (기본: F_11.jpg)')

    # 출력
    parser.add_argument('--out_dir',  default='./results/inhouse')
    parser.add_argument('--no_vis',   action='store_true', help='시각화 저장 생략')
    parser.add_argument('--no_masks', action='store_true', help='마스크 저장 생략')

    # 모델/추론 파라미터
    parser.add_argument('--img_size',         type=int, default=256,
                        help='소형 이미지 리사이즈 해상도')
    parser.add_argument('--patch_size',       type=int, default=256,
                        help='패치 크기 (학습 해상도와 일치 권장)')
    parser.add_argument('--overlap',          type=int, default=64,
                        help='인접 패치 겹침 픽셀 수')
    parser.add_argument('--patch_batch_size', type=int, default=8,
                        help='한 번에 GPU에 올릴 패치 수')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                        help='이진 마스크 저장 threshold (기본: 0.5)')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 장치 설정 ──────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 파일 리스트 ────────────────────────────────────────────────────────────
    print(f"\n파일 리스트 구성 중: {args.input_root}")
    records = build_file_list(
        input_root    = args.input_root,
        brown_root    = args.brown_root,
        red_root      = args.red_root,
        wrinkle_root  = args.wrinkle_root,
        cross_name    = args.cross_name,
        parallel_name = args.parallel_name,
        verbose       = True,
    )

    if not records:
        raise RuntimeError(
            f"처리할 샘플이 없습니다. 경로를 확인하세요: {args.input_root}"
        )

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    print()
    model, cfg = load_model(args.checkpoint, device)
    img_size   = args.img_size or cfg.get('img_size', 256)

    # ── 추론 ──────────────────────────────────────────────────────────────────
    print()
    all_scores = run_inhouse_infer(
        records          = records,
        model            = model,
        out_dir          = out_dir,
        device           = device,
        img_size         = img_size,
        patch_size       = args.patch_size,
        overlap          = args.overlap,
        patch_batch_size = args.patch_batch_size,
        save_vis         = not args.no_vis,
        save_masks       = not args.no_masks,
        mask_threshold   = args.mask_threshold,
    )

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    scores_path = out_dir / 'scores.json'
    _save_scores_json(all_scores, scores_path)
    print(f"\n스코어 저장: {scores_path}")

    has_dice = any(
        s['brown_dice'] is not None or
        s['red_dice']   is not None or
        s['wrinkle_dice'] is not None
        for s in all_scores
    )
    if has_dice:
        dice_path = out_dir / 'dice_scores.txt'
        _save_dice_summary(all_scores, dice_path)
        print(f"Dice 요약 저장: {dice_path}")

    _print_summary(all_scores)


if __name__ == '__main__':
    main()
