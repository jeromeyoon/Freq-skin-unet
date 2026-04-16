"""
skin_infer.py
=============
SkinAnalyzer 추론 스크립트

기능
----
  1. 단일 이미지 쌍 추론 (4K 포함)
       - 이미지가 patch_size보다 크면 자동으로 슬라이딩 윈도우 패치 추론
       - Hann 가중치로 패치 경계 부드럽게 합성
       - 출력 마스크는 원본 해상도 유지
  2. 디렉토리 일괄 추론 (data_prep.py로 사전 패치된 256×256 데이터)
  3. 결과 시각화 저장 (입력 | brown | red | wrinkle 가로 나열)
  4. 스코어 JSON 출력
  5. GT 마스크가 있을 때 Dice score 계산 → 텍스트 파일 저장

실행 예시
---------
  # 4K 단일 이미지 (자동 패치 추론)
  python skin_infer.py \\
      --cross  ./samples/4k_cross.png \\
      --parallel ./samples/4k_parallel.png \\
      --checkpoint ./checkpoints/best_skin_analyzer.pth \\
      --out_dir ./results

  # 패치 파라미터 명시
  python skin_infer.py \\
      --cross  ./samples/4k_cross.png \\
      --parallel ./samples/4k_parallel.png \\
      --checkpoint ./checkpoints/best_skin_analyzer.pth \\
      --out_dir ./results \\
      --patch_size 256 --overlap 64 --patch_batch_size 8

  # 단일 + GT Dice 계산
  python skin_infer.py \\
      --cross  ./samples/cross.png \\
      --parallel ./samples/parallel.png \\
      --gt_brown   ./gt/brown.png \\
      --gt_red     ./gt/red.png \\
      --gt_wrinkle ./gt/wrinkle.png \\
      --checkpoint ./checkpoints/best_skin_analyzer.pth \\
      --out_dir ./results

  # 디렉토리 일괄 추론 (data_prep.py 사전 패치 데이터)
  python skin_infer.py \\
      --input_dir ./patches \\
      --checkpoint ./checkpoints/best_skin_analyzer.pth \\
      --out_dir ./results \\
      --batch_size 8
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from skin_net import build_analyzer, SkinAnalyzer


# ══════════════════════════════════════════════════════════════════════════════
# 이미지 로드 유틸
# ══════════════════════════════════════════════════════════════════════════════
def load_rgb(path: Path, img_size: int) -> torch.Tensor:
    """RGB 이미지 로드 + 리사이즈 → [3, H, W] 텐서 (0~1)"""
    img = Image.open(path).convert('RGB').resize(
        (img_size, img_size), Image.BILINEAR)
    return TF.to_tensor(img)


def load_gray(path: Path, img_size: int) -> torch.Tensor:
    """Grayscale 이미지 로드 + 리사이즈 → [1, H, W] 텐서 (0~1)"""
    img = Image.open(path).convert('L').resize(
        (img_size, img_size), Image.BILINEAR)
    return TF.to_tensor(img)


def load_rgb_full(path: Path) -> torch.Tensor:
    """RGB 이미지 원본 해상도 로드 → [3, H, W] 텐서 (0~1)"""
    return TF.to_tensor(Image.open(path).convert('RGB'))


def load_gray_full(path: Path) -> torch.Tensor:
    """Grayscale 이미지 원본 해상도 로드 → [1, H, W] 텐서 (0~1)"""
    return TF.to_tensor(Image.open(path).convert('L'))


# ══════════════════════════════════════════════════════════════════════════════
# Dice Score 계산
# ══════════════════════════════════════════════════════════════════════════════
def compute_dice(pred_prob: torch.Tensor,
                 gt_mask:   torch.Tensor,
                 threshold: float = 0.5,
                 smooth:    float = 1.0) -> float:
    """
    Binary Dice coefficient (평가용).

    pred_prob : [1, H, W]  sigmoid 적용된 확률맵 (0~1) → threshold로 이진화
    gt_mask   : [1, H, W]  GT binary mask (0.0 or 1.0)  → 이미 이진이므로 round()
    """
    pred_bin = (pred_prob >= threshold).float()
    gt_bin   = gt_mask.round()                   # 0.0/1.0 — float 오차 방어
    inter    = (pred_bin * gt_bin).sum()
    dice     = (2.0 * inter + smooth) / (pred_bin.sum() + gt_bin.sum() + smooth)
    return dice.item()


# ══════════════════════════════════════════════════════════════════════════════
# 모델 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_model(checkpoint_path: str, device: torch.device) -> tuple[SkinAnalyzer, dict]:
    """체크포인트에서 모델 로드. (eval 모드)"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get('cfg', {})

    model = build_analyzer(
        base_ch = cfg.get('base_ch', 64),
        low_r   = cfg.get('low_r',   0.1),
        high_r  = cfg.get('high_r',  0.4),
    ).to(device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    val   = ckpt.get('val_loss', {})
    print(f"체크포인트 로드 완료  epoch={epoch}  "
          f"val_loss={val.get('total', '?'):.4f}")
    return model, cfg


# ══════════════════════════════════════════════════════════════════════════════
# 시각화 저장
# ══════════════════════════════════════════════════════════════════════════════
def save_visualization(
    rgb_cross:    torch.Tensor,
    brown_prob:   torch.Tensor,
    red_prob:     torch.Tensor,
    wrinkle_prob: torch.Tensor,
    save_path:    Path,
    brown_score:  float,
    red_score:    float,
    wrinkle_score: float,
    max_vis_size: int = 1024,
) -> None:
    """
    [입력 | brown | red | wrinkle] 가로 격자 이미지 저장.
    4K 등 고해상도 입력은 max_vis_size로 다운스케일하여 저장.
    """
    H = rgb_cross.shape[1]
    if H > max_vis_size:
        scale     = max_vis_size / H
        new_H     = max_vis_size
        new_W     = max(1, int(rgb_cross.shape[2] * scale))
        rgb_cross    = F.interpolate(rgb_cross.unsqueeze(0),    (new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
        brown_prob   = F.interpolate(brown_prob.unsqueeze(0),   (new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
        red_prob     = F.interpolate(red_prob.unsqueeze(0),     (new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
        wrinkle_prob = F.interpolate(wrinkle_prob.unsqueeze(0), (new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)

    inp_vis     = rgb_cross.clamp(0, 1)
    brown_vis   = brown_prob.repeat(3, 1, 1)
    red_vis     = red_prob.repeat(3, 1, 1)
    wrinkle_vis = wrinkle_prob.repeat(3, 1, 1)

    grid = vutils.make_grid(
        torch.stack([inp_vis, brown_vis, red_vis, wrinkle_vis]),
        nrow=4, padding=2, normalize=False,
    )
    TF.to_pil_image(grid).save(save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 패치 추론 유틸 (4K 등 대용량 이미지용)
# ══════════════════════════════════════════════════════════════════════════════
def _patch_positions(H: int, W: int,
                     patch_size: int, stride: int) -> list[tuple[int, int]]:
    """
    슬라이딩 윈도우 패치 좌상단 (y, x) 좌표 목록 반환.
    마지막 패치가 이미지 경계를 벗어나지 않도록 위치 조정.
    """
    def positions_1d(length: int) -> list[int]:
        pos = list(range(0, length - patch_size, stride))
        # 끝 부분이 덜 채워진 경우 마지막 패치를 경계에 붙임
        if not pos or pos[-1] + patch_size < length:
            pos.append(max(0, length - patch_size))
        return pos

    ys = positions_1d(H)
    xs = positions_1d(W)
    return [(y, x) for y in ys for x in xs]


def _hann_weight(patch_size: int) -> torch.Tensor:
    """
    패치 블렌딩용 2D Hann(코사인) 가중치 [patch_size, patch_size].

    중앙에서 최대, 가장자리에서 0 → 패치 경계의 이음새 제거.
    periodic=False: 양 끝이 정확히 0이 되는 대칭 윈도우.
    """
    w = torch.hann_window(patch_size, periodic=False)
    return (w.unsqueeze(0) * w.unsqueeze(1)).clamp(min=1e-6)  # [P, P]


@torch.no_grad()
def infer_patch_based(
    model:        SkinAnalyzer,
    rgb_cross:    torch.Tensor,    # [3, H, W] CPU, 0~1
    rgb_parallel: torch.Tensor,    # [3, H, W] CPU, 0~1
    skin_mask:    torch.Tensor,    # [1, H, W] CPU, 0~1
    patch_size:   int,
    overlap:      int,
    batch_size:   int,
    device:       torch.device,
) -> dict:
    """
    슬라이딩 윈도우 패치 추론.

    원본 해상도 이미지를 patch_size × patch_size 패치로 분할 →
    배치 단위 모델 추론 → Hann 가중치로 합성 → 원본 해상도 마스크 반환.

    Parameters
    ----------
    patch_size : 모델 학습 해상도 (256)
    overlap    : 인접 패치 간 겹치는 픽셀 수 (권장 64)
    batch_size : 한 번에 GPU에 올릴 패치 수
    """
    _, H, W = rgb_cross.shape
    stride   = patch_size - overlap
    positions = _patch_positions(H, W, patch_size, stride)
    n_patches = len(positions)

    # 출력 누적 텐서 (CPU)
    brown_acc   = torch.zeros(1, H, W)
    red_acc     = torch.zeros(1, H, W)
    wrinkle_acc = torch.zeros(1, H, W)
    weight_acc  = torch.zeros(1, H, W)

    blend_w = _hann_weight(patch_size)          # [P, P]
    blend_w_3d = blend_w.unsqueeze(0)           # [1, P, P]

    pbar = tqdm(range(0, n_patches, batch_size),
                desc=f'  패치 추론 ({n_patches}개)',
                unit='batch', leave=False, dynamic_ncols=True)

    for batch_start in pbar:
        batch_pos = positions[batch_start:batch_start + batch_size]

        # 패치 추출 [B, C, P, P]
        cross_patches    = torch.stack([
            rgb_cross[:, y:y+patch_size, x:x+patch_size] for y, x in batch_pos])
        parallel_patches = torch.stack([
            rgb_parallel[:, y:y+patch_size, x:x+patch_size] for y, x in batch_pos])
        mask_patches     = torch.stack([
            skin_mask[:, y:y+patch_size, x:x+patch_size] for y, x in batch_pos])

        # GPU 추론
        result = model(
            cross_patches.to(device),
            parallel_patches.to(device),
            mask_patches.to(device),
        )

        brown_p   = torch.sigmoid(result.brown_mask).cpu()    # [B, 1, P, P]
        red_p     = torch.sigmoid(result.red_mask).cpu()
        wrinkle_p = torch.sigmoid(result.wrinkle_mask).cpu()

        # Hann 가중치로 누적
        for i, (y, x) in enumerate(batch_pos):
            brown_acc  [:, y:y+patch_size, x:x+patch_size] += brown_p[i]   * blend_w_3d
            red_acc    [:, y:y+patch_size, x:x+patch_size] += red_p[i]     * blend_w_3d
            wrinkle_acc[:, y:y+patch_size, x:x+patch_size] += wrinkle_p[i] * blend_w_3d
            weight_acc [:, y:y+patch_size, x:x+patch_size] += blend_w_3d

    pbar.close()

    # 가중치 정규화
    brown_final   = (brown_acc   / weight_acc).clamp(0, 1)
    red_final     = (red_acc     / weight_acc).clamp(0, 1)
    wrinkle_final = (wrinkle_acc / weight_acc).clamp(0, 1)

    # 스코어: 피부 영역 내 평균 확률 × 100
    def _score(prob_map: torch.Tensor) -> float:
        denom = skin_mask.sum().clamp(min=1.0)
        return ((prob_map * skin_mask).sum() / denom * 100).item()

    return {
        'brown_mask'    : brown_final,
        'red_mask'      : red_final,
        'wrinkle_mask'  : wrinkle_final,
        'brown_score'   : _score(brown_final),
        'red_score'     : _score(red_final),
        'wrinkle_score' : _score(wrinkle_final),
        'rgb_cross'     : rgb_cross,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 단일 추론
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_single(
    model:            SkinAnalyzer,
    cross_path:       Path,
    parallel_path:    Path,
    img_size:         int,
    device:           torch.device,
    skin_mask:        torch.Tensor | None = None,   # [1, H, W] 사전 생성 마스크
    mask_path:        Path | None         = None,   # 파일에서 로드 (skin_mask 없을 때)
    patch_size:       int  = 256,
    overlap:          int  = 64,
    patch_batch_size: int  = 8,
) -> dict:
    """
    단일 이미지 쌍 추론.

    마스크 우선순위: skin_mask 텐서 > mask_path 파일 > 전체 영역(ones)

    이미지 크기 > patch_size 이면 자동으로 슬라이딩 윈도우 패치 추론.
    그렇지 않으면 img_size로 리사이즈 후 단일 pass 추론.

    Returns
    -------
    dict:
        brown_mask, red_mask, wrinkle_mask  : [1, H, W] (0~1 확률)
        brown_score, red_score, wrinkle_score : float (0~100)
        rgb_cross                            : [3, H, W]
    """
    # 이미지 원본 크기 확인
    with Image.open(cross_path) as img:
        orig_W, orig_H = img.size

    use_patch = (orig_H > patch_size or orig_W > patch_size)

    if use_patch:
        print(f"  원본 해상도 {orig_W}×{orig_H} → 패치 추론 "
              f"(patch={patch_size}, overlap={overlap})")
        rgb_cross    = load_rgb_full(cross_path)
        rgb_parallel = load_rgb_full(parallel_path)

        if skin_mask is not None:
            resolved_mask = skin_mask
        elif mask_path is not None and mask_path.exists():
            resolved_mask = load_gray_full(mask_path)
        else:
            resolved_mask = torch.ones(1, orig_H, orig_W)

        return infer_patch_based(
            model        = model,
            rgb_cross    = rgb_cross,
            rgb_parallel = rgb_parallel,
            skin_mask    = resolved_mask,
            patch_size   = patch_size,
            overlap      = overlap,
            batch_size   = patch_batch_size,
            device       = device,
        )

    else:
        # 소형 이미지: 단일 pass 추론
        rgb_cross    = load_rgb(cross_path,    img_size).unsqueeze(0).to(device)
        rgb_parallel = load_rgb(parallel_path, img_size).unsqueeze(0).to(device)

        if skin_mask is not None:
            # 모델 입력 크기에 맞게 리사이즈
            mask = F.interpolate(
                skin_mask.unsqueeze(0), size=(img_size, img_size),
                mode='nearest').to(device)
        elif mask_path is not None and mask_path.exists():
            mask = load_gray(mask_path, img_size).unsqueeze(0).to(device)
        else:
            mask = None

        result = model(rgb_cross, rgb_parallel, mask)

        return {
            'brown_mask'    : torch.sigmoid(result.brown_mask[0]).cpu(),
            'red_mask'      : torch.sigmoid(result.red_mask[0]).cpu(),
            'wrinkle_mask'  : torch.sigmoid(result.wrinkle_mask[0]).cpu(),
            'brown_score'   : result.brown_score[0].item(),
            'red_score'     : result.red_score[0].item(),
            'wrinkle_score' : result.wrinkle_score[0].item(),
            'rgb_cross'     : rgb_cross[0].cpu(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 배치 추론 (디렉토리 — data_prep.py로 사전 패치된 데이터)
# ══════════════════════════════════════════════════════════════════════════════
class InferDataset(torch.utils.data.Dataset):
    """
    디렉토리 기반 추론 데이터셋 (256×256 사전 패치 데이터).

    input_dir/
      rgb_cross/    {stem}.png
      rgb_parallel/ {stem}.png
      mask/         {stem}.png  (optional)
      gt_brown/     {stem}.png  (optional)
      gt_red/       {stem}.png  (optional)
      gt_wrinkle/   {stem}.png  (optional)
    """

    def __init__(self, input_dir: Path, img_size: int):
        self.input_dir = input_dir
        self.img_size  = img_size

        cross_dir = input_dir / 'rgb_cross'
        if not cross_dir.exists():
            raise FileNotFoundError(
                f"rgb_cross 폴더를 찾을 수 없습니다: {cross_dir}\n"
                "디렉토리 구조: input_dir/rgb_cross/, input_dir/rgb_parallel/"
            )

        self.stems = sorted(p.stem for p in cross_dir.glob('*.png'))
        if not self.stems:
            raise RuntimeError(f"rgb_cross 폴더에 .png 파일이 없습니다: {cross_dir}")

        self.has_gt_brown   = (input_dir / 'gt_brown').exists()
        self.has_gt_red     = (input_dir / 'gt_red').exists()
        self.has_gt_wrinkle = (input_dir / 'gt_wrinkle').exists()

        gt_info = [t for t in ('gt_brown', 'gt_red', 'gt_wrinkle')
                   if getattr(self, f'has_{t}')]
        print(f"추론 대상: {len(self.stems)}장")
        if gt_info:
            print(f"GT 폴더 감지: {', '.join(gt_info)} → Dice score 계산 활성화")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        rgb_cross    = load_rgb(
            self.input_dir / 'rgb_cross'    / f'{stem}.png', self.img_size)
        rgb_parallel = load_rgb(
            self.input_dir / 'rgb_parallel' / f'{stem}.png', self.img_size)

        mask_path = self.input_dir / 'mask' / f'{stem}.png'
        mask = load_gray(mask_path, self.img_size) if mask_path.exists() \
               else torch.ones(1, self.img_size, self.img_size)

        item = {'rgb_cross': rgb_cross, 'rgb_parallel': rgb_parallel,
                'mask': mask, 'stem': stem}

        for tag in ('brown', 'red', 'wrinkle'):
            gt_path = self.input_dir / f'gt_{tag}' / f'{stem}.png'
            if gt_path.exists():
                item[f'gt_{tag}']     = load_gray(gt_path, self.img_size)
                item[f'has_gt_{tag}'] = True
            else:
                item[f'gt_{tag}']     = torch.zeros(1, self.img_size, self.img_size)
                item[f'has_gt_{tag}'] = False

        return item


def _infer_collate(batch: list) -> dict:
    return {
        'rgb_cross'      : torch.stack([b['rgb_cross']    for b in batch]),
        'rgb_parallel'   : torch.stack([b['rgb_parallel'] for b in batch]),
        'mask'           : torch.stack([b['mask']         for b in batch]),
        'stem'           : [b['stem'] for b in batch],
        'gt_brown'       : torch.stack([b['gt_brown']     for b in batch]),
        'gt_red'         : torch.stack([b['gt_red']       for b in batch]),
        'gt_wrinkle'     : torch.stack([b['gt_wrinkle']   for b in batch]),
        'has_gt_brown'   : [b['has_gt_brown']   for b in batch],
        'has_gt_red'     : [b['has_gt_red']     for b in batch],
        'has_gt_wrinkle' : [b['has_gt_wrinkle'] for b in batch],
    }


@torch.no_grad()
def infer_directory(
    model:       SkinAnalyzer,
    input_dir:   Path,
    out_dir:     Path,
    img_size:    int,
    batch_size:  int,
    device:      torch.device,
    num_workers: int = 4,
) -> list[dict]:
    """디렉토리 일괄 추론 (256×256 사전 패치 데이터)."""
    ds = InferDataset(input_dir, img_size)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        collate_fn=_infer_collate,
    )

    vis_dir  = out_dir / 'visualizations'
    mask_dir = out_dir / 'masks'
    vis_dir.mkdir(parents=True, exist_ok=True)
    for tag in ('brown', 'red', 'wrinkle'):
        (mask_dir / tag).mkdir(parents=True, exist_ok=True)

    all_scores = []
    pbar = tqdm(loader, desc='추론 중', unit='batch', dynamic_ncols=True)

    for batch in pbar:
        rgb_cross    = batch['rgb_cross'].to(device)
        rgb_parallel = batch['rgb_parallel'].to(device)
        mask         = batch['mask'].to(device)
        stems        = batch['stem']

        gt_brown, gt_red, gt_wrinkle = batch['gt_brown'], batch['gt_red'], batch['gt_wrinkle']
        has_gt_brown   = batch['has_gt_brown']
        has_gt_red     = batch['has_gt_red']
        has_gt_wrinkle = batch['has_gt_wrinkle']

        result = model(rgb_cross, rgb_parallel, mask)

        for i, stem in enumerate(stems):
            brown_prob   = torch.sigmoid(result.brown_mask[i]).cpu()
            red_prob     = torch.sigmoid(result.red_mask[i]).cpu()
            wrinkle_prob = torch.sigmoid(result.wrinkle_mask[i]).cpu()

            b_score = result.brown_score[i].item()
            r_score = result.red_score[i].item()
            w_score = result.wrinkle_score[i].item()

            b_dice = compute_dice(brown_prob,   gt_brown[i])   if has_gt_brown[i]   else None
            r_dice = compute_dice(red_prob,     gt_red[i])     if has_gt_red[i]     else None
            w_dice = compute_dice(wrinkle_prob, gt_wrinkle[i]) if has_gt_wrinkle[i] else None

            save_visualization(
                rgb_cross=rgb_cross[i].cpu(), brown_prob=brown_prob,
                red_prob=red_prob, wrinkle_prob=wrinkle_prob,
                save_path=vis_dir / f'{stem}_vis.png',
                brown_score=b_score, red_score=r_score, wrinkle_score=w_score,
            )

            TF.to_pil_image((brown_prob   >= 0.5).float()).save(mask_dir / 'brown'   / f'{stem}.png')
            TF.to_pil_image((red_prob     >= 0.5).float()).save(mask_dir / 'red'     / f'{stem}.png')
            TF.to_pil_image((wrinkle_prob >= 0.5).float()).save(mask_dir / 'wrinkle' / f'{stem}.png')

            all_scores.append({
                'stem'         : stem,
                'brown_score'  : round(b_score, 4),
                'red_score'    : round(r_score, 4),
                'wrinkle_score': round(w_score, 4),
                'brown_dice'   : round(b_dice, 4) if b_dice is not None else None,
                'red_dice'     : round(r_dice, 4) if r_dice is not None else None,
                'wrinkle_dice' : round(w_dice, 4) if w_dice is not None else None,
            })

        pbar.set_postfix(
            brown=f"{result.brown_score.mean().item():.1f}",
            red=f"{result.red_score.mean().item():.1f}",
            wrinkle=f"{result.wrinkle_score.mean().item():.1f}",
        )

    return all_scores


def _save_dice_txt_batch(all_scores: list[dict], save_path: Path) -> None:
    """배치 추론 결과 Dice score 텍스트 저장."""
    col_w = max((len(s['stem']) for s in all_scores), default=10) + 2
    lines = [
        '# Dice Scores (binary Dice coefficient, threshold=0.5)',
        f"# {'stem':<{col_w}}  {'brown_dice':>12}  {'red_dice':>12}  {'wrinkle_dice':>12}",
        '#' + '-' * (col_w + 44),
    ]
    for s in all_scores:
        b = f"{s['brown_dice']:.4f}"   if s['brown_dice']   is not None else 'N/A'
        r = f"{s['red_dice']:.4f}"     if s['red_dice']     is not None else 'N/A'
        w = f"{s['wrinkle_dice']:.4f}" if s['wrinkle_dice'] is not None else 'N/A'
        lines.append(f"  {s['stem']:<{col_w}}  {b:>12}  {r:>12}  {w:>12}")

    b_vals = [s['brown_dice']   for s in all_scores if s['brown_dice']   is not None]
    r_vals = [s['red_dice']     for s in all_scores if s['red_dice']     is not None]
    w_vals = [s['wrinkle_dice'] for s in all_scores if s['wrinkle_dice'] is not None]
    lines += ['#' + '-' * (col_w + 44), '# 평균 (GT 있는 샘플만)']
    lines.append(f"#   mean_brown_dice   : {sum(b_vals)/len(b_vals):.4f}  (n={len(b_vals)})" if b_vals else "#   mean_brown_dice   : N/A")
    lines.append(f"#   mean_red_dice     : {sum(r_vals)/len(r_vals):.4f}  (n={len(r_vals)})" if r_vals else "#   mean_red_dice     : N/A")
    lines.append(f"#   mean_wrinkle_dice : {sum(w_vals)/len(w_vals):.4f}  (n={len(w_vals)})" if w_vals else "#   mean_wrinkle_dice : N/A")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='SkinAnalyzer 추론 스크립트 (4K 패치 추론 지원)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='학습된 체크포인트 경로 (.pth)')
    parser.add_argument('--out_dir',     type=str, default='./results')
    parser.add_argument('--img_size',    type=int, default=256,
                        help='소형 이미지 리사이즈 해상도 (기본: 256)')
    parser.add_argument('--device',      type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])

    # 패치 추론 파라미터
    patch = parser.add_argument_group(
        '패치 추론 (4K 등 대용량 이미지 자동 적용)')
    patch.add_argument('--patch_size',       type=int, default=256,
                       help='패치 크기 — 학습 해상도와 일치 권장 (기본: 256)')
    patch.add_argument('--overlap',          type=int, default=64,
                       help='인접 패치 간 겹치는 픽셀 수 (기본: 64)')
    patch.add_argument('--patch_batch_size', type=int, default=8,
                       help='한 번에 GPU에 올릴 패치 수 (기본: 8)')

    # 단일 이미지 모드
    single = parser.add_argument_group('단일 이미지 모드')
    single.add_argument('--cross',      type=str, default=None)
    single.add_argument('--parallel',   type=str, default=None)
    single.add_argument('--mask',       type=str, default=None)
    single.add_argument('--gt_brown',   type=str, default=None)
    single.add_argument('--gt_red',     type=str, default=None)
    single.add_argument('--gt_wrinkle', type=str, default=None)

    # 디렉토리 일괄 모드
    batch = parser.add_argument_group('디렉토리 일괄 추론 모드')
    batch.add_argument('--input_dir',   type=str, default=None)
    batch.add_argument('--batch_size',  type=int, default=8)
    batch.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
             if args.device == 'auto' else torch.device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.checkpoint, device)
    img_size   = args.img_size or cfg.get('img_size', 256)

    # ── 단일 이미지 모드 ──────────────────────────────────────────────────────
    if args.cross is not None and args.parallel is not None:
        cross_path    = Path(args.cross)
        parallel_path = Path(args.parallel)
        mask_path     = Path(args.mask) if args.mask else None

        if not cross_path.exists():
            raise FileNotFoundError(f"교차 편광 이미지를 찾을 수 없습니다: {cross_path}")
        if not parallel_path.exists():
            raise FileNotFoundError(f"평행 편광 이미지를 찾을 수 없습니다: {parallel_path}")

        print(f"\n단일 추론: {cross_path.name}")
        pred = infer_single(
            model            = model,
            cross_path       = cross_path,
            parallel_path    = parallel_path,
            img_size         = img_size,
            device           = device,
            mask_path        = mask_path,
            patch_size       = args.patch_size,
            overlap          = args.overlap,
            patch_batch_size = args.patch_batch_size,
        )

        stem     = cross_path.stem
        vis_path = out_dir / f'{stem}_vis.png'
        save_visualization(
            rgb_cross=pred['rgb_cross'], brown_prob=pred['brown_mask'],
            red_prob=pred['red_mask'],   wrinkle_prob=pred['wrinkle_mask'],
            save_path=vis_path,
            brown_score=pred['brown_score'],
            red_score=pred['red_score'],
            wrinkle_score=pred['wrinkle_score'],
        )

        # 원본 해상도 마스크 저장 (이진화)
        TF.to_pil_image((pred['brown_mask']   >= 0.5).float()).save(out_dir / f'{stem}_brown.png')
        TF.to_pil_image((pred['red_mask']     >= 0.5).float()).save(out_dir / f'{stem}_red.png')
        TF.to_pil_image((pred['wrinkle_mask'] >= 0.5).float()).save(out_dir / f'{stem}_wrinkle.png')

        scores = {
            'stem'         : stem,
            'brown_score'  : round(pred['brown_score'],   2),
            'red_score'    : round(pred['red_score'],     2),
            'wrinkle_score': round(pred['wrinkle_score'], 2),
        }
        print(f"\n결과 스코어:")
        print(f"  갈색 반점(brown)  : {scores['brown_score']:6.2f} / 100")
        print(f"  적색 반점(red)    : {scores['red_score']:6.2f} / 100")
        print(f"  주름 (wrinkle)    : {scores['wrinkle_score']:6.2f} / 100")

        score_path = out_dir / f'{stem}_scores.json'
        with open(score_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

        # GT Dice 계산
        gt_paths = {
            'brown'  : Path(args.gt_brown)   if args.gt_brown   else None,
            'red'    : Path(args.gt_red)     if args.gt_red     else None,
            'wrinkle': Path(args.gt_wrinkle) if args.gt_wrinkle else None,
        }
        has_any_gt = any(p is not None and p.exists() for p in gt_paths.values())

        if has_any_gt:
            H, W = pred['brown_mask'].shape[1], pred['brown_mask'].shape[2]
            dice_lines = [f'# Dice Scores — {stem}',
                          '# (binary Dice coefficient, threshold=0.5)', '#']
            print(f"\nDice Scores (binary, threshold=0.5):")

            for tag, pred_key in [('brown', 'brown_mask'),
                                   ('red',   'red_mask'),
                                   ('wrinkle', 'wrinkle_mask')]:
                gt_p = gt_paths[tag]
                if gt_p is not None and gt_p.exists():
                    # GT도 예측 마스크와 동일 해상도로 로드
                    with Image.open(gt_p) as _img:
                        gW, gH = _img.size
                    gt_tensor = load_gray_full(gt_p)
                    # 해상도 불일치 시 NEAREST 리사이즈 (binary mask 경계값 보호)
                    if gt_tensor.shape[1] != H or gt_tensor.shape[2] != W:
                        gt_tensor = F.interpolate(
                            gt_tensor.unsqueeze(0), (H, W),
                            mode='nearest').squeeze(0)
                    dice_val = compute_dice(pred[pred_key], gt_tensor)
                    dice_lines.append(
                        f"  {tag}_dice    : {dice_val:.4f}  (GT: {gt_p.name})")
                    print(f"  {tag}_dice    : {dice_val:.4f}  (GT: {gt_p.name})")
                else:
                    dice_lines.append(f"  {tag}_dice    : N/A     (GT 없음)")

            dice_path = out_dir / f'{stem}_dice.txt'
            with open(dice_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(dice_lines) + '\n')
            print(f"  → {dice_path}")

        print(f"\n저장 완료:")
        print(f"  시각화  : {vis_path}")
        print(f"  스코어  : {score_path}")
        mask_H, mask_W = pred['brown_mask'].shape[1], pred['brown_mask'].shape[2]
        print(f"  마스크 해상도: {mask_W}×{mask_H}")

    # ── 디렉토리 일괄 추론 모드 ───────────────────────────────────────────────
    elif args.input_dir is not None:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")

        print(f"\n일괄 추론: {input_dir}")
        all_scores = infer_directory(
            model=model, input_dir=input_dir, out_dir=out_dir,
            img_size=img_size, batch_size=args.batch_size,
            device=device, num_workers=args.num_workers,
        )

        scores_path = out_dir / 'scores.json'
        with open(scores_path, 'w', encoding='utf-8') as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)

        has_dice = any(
            s['brown_dice'] is not None or s['red_dice'] is not None or
            s['wrinkle_dice'] is not None for s in all_scores)
        if has_dice:
            dice_path = out_dir / 'dice_scores.txt'
            _save_dice_txt_batch(all_scores, dice_path)

        if all_scores:
            n = len(all_scores)
            avg_b = sum(s['brown_score']   for s in all_scores) / n
            avg_r = sum(s['red_score']     for s in all_scores) / n
            avg_w = sum(s['wrinkle_score'] for s in all_scores) / n
            print(f"\n추론 완료  총 {n}장")
            print(f"  갈색 반점 평균: {avg_b:.2f}")
            print(f"  적색 반점 평균: {avg_r:.2f}")
            print(f"  주름 평균     : {avg_w:.2f}")
            print(f"  스코어 저장  : {scores_path}")
            if has_dice:
                b_vals = [s['brown_dice']   for s in all_scores if s['brown_dice']   is not None]
                r_vals = [s['red_dice']     for s in all_scores if s['red_dice']     is not None]
                w_vals = [s['wrinkle_dice'] for s in all_scores if s['wrinkle_dice'] is not None]
                print(f"\nDice 요약 (binary, threshold=0.5):")
                if b_vals: print(f"  mean_brown_dice   : {sum(b_vals)/len(b_vals):.4f}  (n={len(b_vals)})")
                if r_vals: print(f"  mean_red_dice     : {sum(r_vals)/len(r_vals):.4f}  (n={len(r_vals)})")
                if w_vals: print(f"  mean_wrinkle_dice : {sum(w_vals)/len(w_vals):.4f}  (n={len(w_vals)})")
                print(f"  Dice 저장        : {dice_path}")

    else:
        raise ValueError(
            "추론 모드를 지정하세요.\n"
            "  단일: --cross <path> --parallel <path>\n"
            "  일괄: --input_dir <dir>"
        )


if __name__ == '__main__':
    main()
