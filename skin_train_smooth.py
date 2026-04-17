"""
skin_train_smooth.py
====================
SkinAnalyzer 학습 스크립트 — stable/smooth 개선 버전

skin_train.py 대비 변경사항:
  - ir_encoder_stable 사용 (OD 수치 안정화 + FrequencyGate 조기 억제 강화)
  - skin_loss_smooth 사용 (Phase 전환 warm-up + 초기 freq_reg 보호)
  - torch.load weights_only=True (보안)
  - best_val 저장 순서 수정 (last_checkpoint에도 갱신된 best_val 기록)
  - ZeroDivisionError 방지: len(loader) → max(len(loader), 1)

실행
----
  # 패치 먼저 준비
  python data_prep.py \
      --input_path ./raw_data/images \
      --gt_path    ./raw_data/gt \
      --patch_dir  ./patches \
      --patch_size 256 --stride 256

  # 학습
  python skin_train_smooth.py

  # 이어서 학습
  python skin_train_smooth.py --resume ./checkpoints/last_checkpoint.pth
"""

# ir_encoder_stable을 ir_encoder로 주입 (skin_net.py 수정 없이 교체)
import sys
import ir_encoder_stable as _ir_stable
sys.modules['ir_encoder'] = _ir_stable

import argparse
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from pathlib import Path

import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from tqdm import tqdm

from skin_net          import build_analyzer
from skin_loss_smooth  import SkinAnalyzerLoss, get_loss_weights
from skin_dataset      import SkinDataset, skin_collate_fn, ExcludedDataset, preview_collate_fn
from ambient_aug       import apply_batch_illumination_aug


# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    patch_dir      = './patches',
    checkpoint_dir = './checkpoints_smooth',   # 원본 checkpoints와 분리

    base_ch        = 64,
    low_r          = 0.1,
    high_r         = 0.4,
    img_size       = 256,

    epochs         = 100,
    batch_size     = 16,
    lr             = 1e-4,
    weight_decay   = 1e-4,
    num_workers    = 2,
    val_ratio      = 0.1,

    w_brown        = 1.0,
    w_red          = 1.0,
    w_wrinkle      = 1.0,
    w_recon        = 0.3,
    # w_consist / w_freq_reg 는 get_loss_weights 스케줄로 자동 제어

    aug_warmup_epochs = 30,

    use_weighted_sampler = True,
    sampler_neg_weight   = 0.05,

    intensity_range   = (0.6, 1.4),
    color_temp_range  = (0.7, 1.3),
    tint_range        = (0.8, 1.2),
    vignette_prob     = 0.5,
    gradient_prob     = 0.5,

    seed              = 42,

    preview_interval  = 10,
    preview_n_images  = 8,
    preview_dir       = './previews_smooth',
)


# ══════════════════════════════════════════════════════════════════════════════
# 학습 제외 이미지 예측 결과 시각화
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def save_preview(model, preview_loader, device, epoch, save_dir: Path, n_images: int):
    model.eval()
    save_dir = save_dir / f'epoch_{epoch:03d}'
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for batch in preview_loader:
        if saved >= n_images:
            break

        rgb_cross    = batch['rgb_cross'].to(device)
        rgb_parallel = batch['rgb_parallel'].to(device)
        mask         = batch['mask'].to(device)
        stems        = batch['stem']

        result = model(rgb_cross, rgb_parallel, mask)

        for i in range(rgb_cross.shape[0]):
            if saved >= n_images:
                break

            inp   = rgb_cross[i].cpu().clamp(0, 1)
            brown = torch.sigmoid(result.brown_mask[i]).cpu().clamp(0, 1)
            red   = torch.sigmoid(result.red_mask[i]).cpu().clamp(0, 1)
            wrk   = torch.sigmoid(result.wrinkle_mask[i]).cpu().clamp(0, 1)

            brown_rgb = brown.repeat(3, 1, 1)
            red_rgb   = red.repeat(3, 1, 1)
            wrk_rgb   = wrk.repeat(3, 1, 1)

            grid = vutils.make_grid(
                torch.stack([inp, brown_rgb, red_rgb, wrk_rgb]),
                nrow=4, padding=2, normalize=False,
            )
            TF.to_pil_image(grid).save(save_dir / f'{stems[i]}.png')
            saved += 1

    if saved:
        print(f"  Preview 저장: {save_dir}  ({saved}장)")


# ══════════════════════════════════════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device, cfg, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    log = {k: 0.0 for k in ['brown', 'red', 'wrinkle', 'recon', 'consist', 'freq_reg']}

    warmup    = cfg.get('aug_warmup_epochs', 0)
    aug_scale = min(1.0, epoch / max(warmup, 1)) if warmup > 0 else 1.0

    def _scale_range(rng, scale):
        mid  = (rng[0] + rng[1]) / 2.0
        half = (rng[1] - rng[0]) / 2.0 * scale
        return (mid - half, mid + half)

    aug_kwargs = dict(
        intensity_range  = _scale_range(cfg['intensity_range'],  aug_scale),
        color_temp_range = _scale_range(cfg['color_temp_range'], aug_scale),
        tint_range       = _scale_range(cfg['tint_range'],       aug_scale),
        vignette_prob    = cfg['vignette_prob']  * aug_scale,
        gradient_prob    = cfg['gradient_prob']  * aug_scale,
    )

    pbar = tqdm(loader,
                desc=f"Train {epoch:03d}/{total_epochs}",
                unit='batch', leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar, 1):
        rgb_cross    = batch['rgb_cross'].to(device)
        rgb_parallel = batch['rgb_parallel'].to(device)
        brown_gt     = batch['brown'].to(device)
        red_gt       = batch['red'].to(device)
        wrinkle_gt   = batch['wrinkle'].to(device)
        mask         = batch['mask'].to(device)

        has_brown   = batch['has_brown']
        has_red     = batch['has_red']
        has_wrinkle = batch['has_wrinkle']

        # 조명 변형 입력으로 주 forward
        rgb_cross_aug    = apply_batch_illumination_aug(rgb_cross,    **aug_kwargs)
        rgb_parallel_aug = apply_batch_illumination_aug(rgb_parallel, **aug_kwargs)
        result = model(rgb_cross_aug, rgb_parallel_aug, mask)

        # Consistency: 원본 입력으로 추가 forward (w_consist > 0 일 때만)
        result_orig = None
        if criterion.w_consist > 0:
            result_orig = model(rgb_cross, rgb_parallel, mask)

        loss, detail = criterion(
            result,
            brown_gt, red_gt, wrinkle_gt,
            rgb_cross   = rgb_cross,
            face_mask   = mask,
            has_brown   = has_brown,
            has_red     = has_red,
            has_wrinkle = has_wrinkle,
            result_aug  = result_orig,
            model       = model if criterion.w_freq_reg > 0 else None,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in log:
            log[k] += detail.get(k, 0.0)

        pbar.set_postfix(
            loss=f"{total_loss / step:.4f}",
            brown=f"{log['brown'] / step:.4f}",
            red=f"{log['red'] / step:.4f}",
        )

    pbar.close()
    n = max(len(loader), 1)
    return {'total': total_loss / n, **{k: v / n for k, v in log.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# 검증
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    log = {k: 0.0 for k in ['brown', 'red', 'wrinkle', 'recon']}

    pbar = tqdm(loader, desc="  Val  ", unit='batch', leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar, 1):
        rgb_cross    = batch['rgb_cross'].to(device)
        rgb_parallel = batch['rgb_parallel'].to(device)
        brown_gt     = batch['brown'].to(device)
        red_gt       = batch['red'].to(device)
        wrinkle_gt   = batch['wrinkle'].to(device)
        mask         = batch['mask'].to(device)

        result = model(rgb_cross, rgb_parallel, mask)
        loss, detail = criterion(
            result,
            brown_gt, red_gt, wrinkle_gt,
            rgb_cross   = rgb_cross,
            face_mask   = mask,
            has_brown   = batch['has_brown'],
            has_red     = batch['has_red'],
            has_wrinkle = batch['has_wrinkle'],
        )

        total_loss += loss.item()
        for k in log:
            log[k] += detail.get(k, 0.0)

        pbar.set_postfix(loss=f"{total_loss / step:.4f}")

    pbar.close()
    n = max(len(loader), 1)
    return {'total': total_loss / n, **{k: v / n for k, v in log.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='SkinAnalyzer stable/smooth 학습')
    parser.add_argument('--resume', type=str, default=None,
                        help='재개할 체크포인트 경로')
    args = parser.parse_args()

    seed = CFG['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Seed: {seed}")
    print(f"Encoder : ir_encoder_stable  (eps=1e-3, OD clamp±6, low_logit=-3)")
    print(f"Loss    : skin_loss_smooth   (Phase warm-up, early freq_reg=0.02)")

    ckpt_dir = Path(CFG['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_analyzer(
        base_ch = CFG['base_ch'],
        low_r   = CFG['low_r'],
        high_r  = CFG['high_r'],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    full_ds  = SkinDataset(CFG['patch_dir'], CFG['img_size'], augment=True)
    val_size = max(1, int(len(full_ds) * CFG['val_ratio']))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])

    val_ds.dataset         = copy.copy(full_ds)
    val_ds.dataset.augment = False

    if CFG.get('use_weighted_sampler', False):
        all_weights  = full_ds.get_sample_weights(
            neg_weight=CFG.get('sampler_neg_weight', 0.05))
        train_weights = [all_weights[i] for i in train_ds.indices]
        sampler = WeightedRandomSampler(
            weights=train_weights, num_samples=len(train_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, CFG['batch_size'], sampler=sampler,
            num_workers=CFG['num_workers'], pin_memory=True,
            persistent_workers=CFG['num_workers'] > 0,
            prefetch_factor=4 if CFG['num_workers'] > 0 else None,
            collate_fn=skin_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_ds, CFG['batch_size'], shuffle=True,
            num_workers=CFG['num_workers'], pin_memory=True,
            persistent_workers=CFG['num_workers'] > 0,
            prefetch_factor=4 if CFG['num_workers'] > 0 else None,
            collate_fn=skin_collate_fn,
        )

    val_loader = DataLoader(
        val_ds, CFG['batch_size'], shuffle=False,
        num_workers=CFG['num_workers'], pin_memory=True,
        persistent_workers=CFG['num_workers'] > 0,
        prefetch_factor=4 if CFG['num_workers'] > 0 else None,
        collate_fn=skin_collate_fn,
    )
    print(f"Train: {len(train_ds)} / Val: {len(val_ds)}")

    # Preview loader
    preview_loader = None
    if CFG['preview_interval'] > 0:
        excl_ds = ExcludedDataset.from_skin_dataset(full_ds)
        if len(excl_ds) > 0:
            preview_loader = DataLoader(
                excl_ds,
                batch_size  = CFG['preview_n_images'],
                shuffle     = True,
                num_workers = 0,
                collate_fn  = preview_collate_fn,
            )
            print(f"Preview 대상: {len(excl_ds)}장  (매 {CFG['preview_interval']} 에폭마다)")

    criterion = SkinAnalyzerLoss(
        w_brown    = CFG['w_brown'],
        w_red      = CFG['w_red'],
        w_wrinkle  = CFG['w_wrinkle'],
        w_recon    = CFG['w_recon'],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float('inf')

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print("  [WARN] optimizer 상태 없음 → 초기화")
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            for _ in range(ckpt['epoch']):
                scheduler.step()
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"Resume: epoch {ckpt['epoch']} → {start_epoch}부터 재개  "
              f"(best_val={best_val:.4f}  lr={scheduler.get_last_lr()[0]:.2e})")
    # ─────────────────────────────────────────────────────────────────────────

    print("\n학습 시작...")

    epoch_bar = tqdm(range(start_epoch, CFG['epochs'] + 1),
                     desc='Epochs', unit='ep', dynamic_ncols=True)

    for epoch in epoch_bar:
        # Loss 가중치 스케줄 적용 (smooth 버전)
        weights = get_loss_weights(
            epoch, CFG['epochs'],
            w_brown   = CFG['w_brown'],
            w_red     = CFG['w_red'],
            w_wrinkle = CFG['w_wrinkle'],
            w_recon   = CFG['w_recon'],
        )
        for k, v in weights.items():
            setattr(criterion, k, v)

        train_log = train_one_epoch(
            model, train_loader, criterion, optimizer, device, CFG,
            epoch, CFG['epochs'],
        )
        val_log = validate(model, val_loader, criterion, device)
        scheduler.step()

        enc = model.cross_encoder
        low_gates = [
            torch.sigmoid(enc.freq1.low_logit).mean().item(),
            torch.sigmoid(enc.freq2.low_logit).mean().item(),
            torch.sigmoid(enc.freq3.low_logit).mean().item(),
        ]

        is_best = val_log['total'] < best_val

        epoch_bar.set_postfix(
            train=f"{train_log['total']:.4f}",
            val=f"{val_log['total']:.4f}",
            brown=f"{val_log['brown']:.4f}",
            red=f"{val_log['red']:.4f}",
        )

        tqdm.write(
            f"[{epoch:03d}/{CFG['epochs']}]"
            f"  trn={train_log['total']:.4f}"
            f"  val={val_log['total']:.4f}"
            f"  B={val_log['brown']:.4f}"
            f"  R={val_log['red']:.4f}"
            f"  W={val_log['wrinkle']:.4f}"
            f"  rc={val_log['recon']:.4f}"
            f"  gate=[{low_gates[0]:.3f},{low_gates[1]:.3f},{low_gates[2]:.3f}]"
            f"  w_consist={criterion.w_consist:.3f}"
            f"  w_freq={criterion.w_freq_reg:.3f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
            + (' ★' if is_best else '')
        )

        # best_val 먼저 갱신 → last_checkpoint에도 최신값 기록
        if is_best:
            best_val = val_log['total']

        last_ckpt = {
            'epoch'                : epoch,
            'state_dict'           : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'best_val'             : best_val,   # 갱신된 값
            'val_loss'             : val_log,
            'cfg'                  : CFG,
        }
        torch.save(last_ckpt, ckpt_dir / 'last_checkpoint.pth')

        if is_best:
            torch.save(last_ckpt, ckpt_dir / 'best_skin_analyzer.pth')
            tqdm.write(f"  ★ Best 저장: val={best_val:.4f}")

        if (preview_loader is not None
                and CFG['preview_interval'] > 0
                and epoch % CFG['preview_interval'] == 0):
            save_preview(
                model, preview_loader, device,
                epoch,
                save_dir = Path(CFG['preview_dir']),
                n_images = CFG['preview_n_images'],
            )

    epoch_bar.close()
    print("\n학습 완료!")


if __name__ == '__main__':
    main()
