"""
Simple UNet inference - 4K 이미지 패치 기반 처리.

패치 단위로 분할 → 배치 추론 → Hann window 가중 합산으로 원본 해상도 복원.

Usage:
    # GT 없음
    python inference.py --checkpoint checkpoints/brown/best_brown.pth \
                        --input_root /path/to/rgb \
                        --save_dir ./results

    # GT 있음 (메트릭 계산)
    python inference.py --checkpoint checkpoints/brown/best_brown.pth \
                        --input_root /path/to/rgb \
                        --gt_root /path/to/masks \
                        --save_dir ./results
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from tqdm import tqdm

from model import EfficientNetUNet
from loss  import SegLoss

# ─────────────────────── constants ──────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS      = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
OVERLAY_COLOR = (255, 64, 64)
OVERLAY_ALPHA = 0.45


# ─────────────────────── patch inference ────────────────────

def _hann2d(size: int) -> torch.Tensor:
    """패치 경계 블렌딩용 2D Hann window [size, size]."""
    h = torch.hann_window(size, periodic=False)
    return h.unsqueeze(1) * h.unsqueeze(0)


def patch_infer(model: torch.nn.Module,
                img_t: torch.Tensor,
                patch_size: int,
                stride: int,
                batch_size: int,
                device: torch.device) -> torch.Tensor:
    """
    img_t     : [3, H, W] normalized tensor (CPU)
    returns   : [1, H, W] prediction in [0,1] (CPU)

    동작:
      1. 이미지를 패치로 분할 (패딩 후)
      2. 배치 단위로 모델 추론
      3. Hann window 가중치로 겹치는 영역 블렌딩
      4. 원본 해상도로 crop 후 반환
    """
    C, H, W = img_t.shape

    # ── 패딩 계산 (stride 단위로 패치가 이미지를 완전히 커버하도록) ──
    def pad_to_fit(size, ps, st):
        if size <= ps:
            return ps - size
        remainder = (size - ps) % st
        return (st - remainder) % st

    pad_h = pad_to_fit(H, patch_size, stride)
    pad_w = pad_to_fit(W, patch_size, stride)

    # reflect padding으로 경계 아티팩트 최소화
    img_pad = F.pad(img_t.unsqueeze(0),
                    (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
    _, H_p, W_p = img_pad.shape

    # ── 패치 좌표 생성 ──
    ys = list(range(0, H_p - patch_size + 1, stride))
    xs = list(range(0, W_p - patch_size + 1, stride))

    patches, coords = [], []
    for y in ys:
        for x in xs:
            patches.append(img_pad[:, y:y+patch_size, x:x+patch_size])
            coords.append((y, x))

    # ── 배치 추론 ──
    weight_map = _hann2d(patch_size).to(device)   # [P,P]
    accum  = torch.zeros(1, H_p, W_p)             # 누적 예측값
    counts = torch.zeros(1, H_p, W_p)             # 누적 가중치

    model.eval()
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch_p = torch.stack(patches[start:start+batch_size]).to(device)
            preds   = model(batch_p)               # [B,1,P,P]

            for j, pred in enumerate(preds):
                y, x = coords[start + j]
                w    = weight_map                  # [P,P]
                accum[0, y:y+patch_size, x:x+patch_size]  += (pred[0] * w).cpu()
                counts[0, y:y+patch_size, x:x+patch_size] += w.cpu()

    # ── 가중 평균 → 원본 크기 crop ──
    pred_full = accum / counts.clamp(min=1e-6)
    return pred_full[:, :H, :W]                   # [1, H, W]


# ─────────────────────── metrics ────────────────────────────

def compute_metrics(pred_bin: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """pred_bin, gt: [1, H, W] binary float"""
    p     = pred_bin.view(-1)
    g     = gt.view(-1)
    inter = (p * g).sum()
    dice  = ((2 * inter + 1) / (p.sum() + g.sum() + 1)).item()
    iou   = ((inter + 1) / (p.sum() + g.sum() - inter + 1)).item()
    return dice, iou


# ─────────────────────── visualisation ──────────────────────

def save_pred_mask(pred_np: np.ndarray, path: Path):
    Image.fromarray((pred_np * 255).astype(np.uint8)).save(path)


def save_overlay(orig: Image.Image, pred_np: np.ndarray, path: Path):
    mask      = Image.fromarray((pred_np * 255).astype(np.uint8), 'L')
    color_lay = Image.new('RGBA', orig.size, OVERLAY_COLOR + (int(255 * OVERLAY_ALPHA),))
    overlay   = Image.new('RGBA', orig.size, (0, 0, 0, 0))
    overlay.paste(color_lay, mask=mask)
    Image.alpha_composite(orig.convert('RGBA'), overlay).convert('RGB').save(path)


def save_compare(orig: Image.Image,
                 gt_pil: Image.Image | None,
                 pred_np: np.ndarray,
                 path: Path,
                 dice: float | None = None,
                 scale: int = 4):
    """
    4K 이미지는 비교 이미지도 너무 크므로 scale 배 축소 후 저장.
    패널: [input | GT | pred]  또는  [input | pred]
    """
    W, H  = orig.size
    W_s   = W // scale
    H_s   = H // scale
    cols  = 3 if gt_pil is not None else 2
    canvas = Image.new('RGB', (W_s * cols, H_s), (30, 30, 30))

    canvas.paste(orig.resize((W_s, H_s), Image.BILINEAR), (0, 0))

    if gt_pil is not None:
        gt_arr = np.array(gt_pil.resize((W_s, H_s), Image.NEAREST))
        canvas.paste(Image.fromarray(np.stack([gt_arr] * 3, -1)), (W_s, 0))

    pred_s = (pred_np * 255).astype(np.uint8)
    pred_s = Image.fromarray(np.stack([pred_s] * 3, -1)).resize((W_s, H_s), Image.NEAREST)
    canvas.paste(pred_s, (W_s * (cols - 1), 0))

    if dice is not None:
        ImageDraw.Draw(canvas).text(
            (W_s * (cols - 1) + 6, 6), f"Dice {dice:.3f}", fill=(255, 220, 0))

    canvas.save(path)


# ─────────────────────── file helpers ───────────────────────

def collect_images(input_root: Path,
                   gt_root: Path | None,
                   img_filename: str = 'F_11.jpg') -> list[tuple[Path, Path | None, str]]:
    """
    input_root/ID/img_filename 구조를 탐색.
    반환: [(img_path, gt_path or None, ID), ...]

    GT 매칭 순서:
      1. gt_root/ID.png
      2. gt_root/ID/{img_filename stem}.png
      3. gt_root/ID/ 안의 첫 번째 이미지
    """
    items = []
    for id_dir in sorted(input_root.iterdir()):
        if not id_dir.is_dir():
            continue

        img_path = id_dir / img_filename
        if not img_path.exists():
            # img_filename이 없으면 폴더 안 첫 번째 이미지 사용
            candidates = [p for p in sorted(id_dir.iterdir())
                          if p.suffix.lower() in IMG_EXTS]
            if not candidates:
                continue
            img_path = candidates[0]

        gt = None
        if gt_root is not None:
            stem = img_path.stem
            id_name = id_dir.name
            # 우선순위 순으로 GT 경로 탐색
            for candidate in [
                gt_root / f"{id_name}.png",
                gt_root / id_name / f"{stem}.png",
                gt_root / id_name / f"{id_name}.png",
            ]:
                if candidate.exists():
                    gt = candidate
                    break
            # 위 세 경로 모두 없으면 gt_root/ID/ 안 첫 번째 이미지
            if gt is None:
                gt_id_dir = gt_root / id_name
                if gt_id_dir.is_dir():
                    found = [p for p in sorted(gt_id_dir.iterdir())
                             if p.suffix.lower() in IMG_EXTS]
                    if found:
                        gt = found[0]

        items.append((img_path, gt, id_dir.name))   # ID = 서브폴더 이름

    assert items, f"No images found under {input_root} (looking for {img_filename})"
    return items


def load_image_tensor(img_path: Path) -> tuple[torch.Tensor, Image.Image]:
    """원본 PIL 이미지와 정규화된 텐서를 함께 반환."""
    pil = Image.open(img_path).convert('RGB')
    t   = TF.normalize(TF.to_tensor(pil), IMAGENET_MEAN, IMAGENET_STD)
    return t, pil                              # [3,H,W], PIL


# ─────────────────────── main ───────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   required=True)
    p.add_argument('--input_root',   required=True)
    p.add_argument('--gt_root',       default=None)
    p.add_argument('--img_filename',  default='F_11.jpg',
                   help='각 ID 폴더 안에서 읽을 이미지 파일명')
    p.add_argument('--save_dir',      default='./inference_results')
    p.add_argument('--patch_size',   type=int,   default=512,
                   help='패치 크기 (모델 학습 img_size와 동일 권장)')
    p.add_argument('--stride',       type=int,   default=384,
                   help='패치 이동 간격. 작을수록 overlap 증가, 품질↑ 속도↓')
    p.add_argument('--batch_size',   type=int,   default=8,
                   help='한 번에 처리할 패치 수')
    p.add_argument('--threshold',    type=float, default=0.5)
    p.add_argument('--compare_scale',type=int,   default=4,
                   help='compare 이미지 축소 배율 (4K → 1/4)')
    p.add_argument('--num_workers',  type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 출력 폴더 ──
    save_dir = Path(args.save_dir)
    pred_dir = save_dir / 'pred'
    over_dir = save_dir / 'overlay'
    comp_dir = save_dir / 'compare'
    for d in [pred_dir, over_dir, comp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ──
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EfficientNetUNet(pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"Checkpoint  : {args.checkpoint}  (epoch {ckpt.get('epoch','?')})")
    print(f"Patch size  : {args.patch_size}  stride: {args.stride}  batch: {args.batch_size}")

    criterion = SegLoss()
    gt_root   = Path(args.gt_root) if args.gt_root else None
    items     = collect_images(Path(args.input_root), gt_root, args.img_filename)
    print(f"Images      : {len(items)}  |  GT: {'yes' if gt_root else 'no'}\n")

    all_dice, all_iou, all_loss = [], [], []
    csv_rows: list[dict] = []

    for img_path, gt_path, id_name in tqdm(items, desc='Images'):
        stem = id_name          # 출력 파일명 = ID 폴더명

        # ── 이미지 로드 ──
        img_t, orig_pil = load_image_tensor(img_path)
        H, W = img_t.shape[1], img_t.shape[2]

        n_patches = (
            len(range(0, max(H, args.patch_size) - args.patch_size + 1, args.stride)) *
            len(range(0, max(W, args.patch_size) - args.patch_size + 1, args.stride))
        )
        tqdm.write(f"  {stem}  {W}×{H}  →  ~{n_patches} patches")

        # ── 패치 추론 ──
        pred_map = patch_infer(model, img_t,
                               args.patch_size, args.stride,
                               args.batch_size, device)        # [1,H,W] CPU
        pred_np  = (pred_map[0].numpy() > args.threshold).astype(np.uint8)

        # ── 저장 ──
        save_pred_mask(pred_np, pred_dir / f"{stem}.png")
        save_overlay(orig_pil, pred_np, over_dir / f"{stem}.png")

        # ── GT 있으면 metric ──
        dice_val = None
        if gt_path is not None:
            gt_pil  = Image.open(gt_path).convert('L')
            gt_arr  = np.array(gt_pil.resize((W, H), Image.NEAREST))
            gt_t    = torch.from_numpy((gt_arr > 127).astype(np.float32)).unsqueeze(0)

            pred_bin = (pred_map > args.threshold).float()
            dice_val, iou_val = compute_metrics(pred_bin, gt_t)
            loss_val = criterion(pred_map.unsqueeze(0), gt_t.unsqueeze(0)).item()

            all_dice.append(dice_val)
            all_iou.append(iou_val)
            all_loss.append(loss_val)
            csv_rows.append({'stem': stem,
                             'dice': f'{dice_val:.4f}',
                             'iou':  f'{iou_val:.4f}',
                             'loss': f'{loss_val:.4f}'})
            tqdm.write(f"    dice={dice_val:.4f}  iou={iou_val:.4f}  loss={loss_val:.4f}")
        else:
            gt_pil = None

        save_compare(orig_pil, gt_pil, pred_np,
                     comp_dir / f"{stem}.png",
                     dice=dice_val, scale=args.compare_scale)

    # ── 요약 ──
    print(f"\n{'='*54}")
    print(f"Results → {save_dir}")
    print(f"  pred/      binary masks  (원본 해상도)")
    print(f"  overlay/   prediction overlay on original")
    print(f"  compare/   side-by-side  (1/{args.compare_scale} 축소)")

    if all_dice:
        print(f"\n── Metrics (n={len(all_dice)}) ───────────────────────────")
        print(f"  Dice  {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
        print(f"  IoU   {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
        print(f"  Loss  {np.mean(all_loss):.4f}")

        csv_path = save_dir / 'metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['stem', 'dice', 'iou', 'loss'])
            w.writeheader()
            w.writerows(csv_rows)
            w.writerow({'stem': 'MEAN',
                        'dice': f'{np.mean(all_dice):.4f}',
                        'iou':  f'{np.mean(all_iou):.4f}',
                        'loss': f'{np.mean(all_loss):.4f}'})
        print(f"  CSV   {csv_path}")


if __name__ == '__main__':
    main()
