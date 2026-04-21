"""
skin_train_v2.py
================
SkinAnalyzerV2 학습 스크립트 — H1 + H2 수정 버전

skin_train_smooth.py 대비 변경사항:
  - skin_net_v2 사용 (SkinAnalyzerV2: HueBranch + SpotHeadV2 + ODFrequencyFilter)
  - sys.modules 주입 불필요 (ir_encoder_v2는 독립 구현)
  - gate 로깅: od_filter.low_logit (R/G/B 채널별 3값)
  - freq_reg: ODFrequencyFilter.low_logit 직접 정규화
              (기존 freq1.low_logit 기반 _freq_reg와 인터페이스 달라 직접 계산)

실행
----
  # 패치 먼저 준비
  python data_prep.py \
      --input_path ./raw_data/images \
      --gt_path    ./raw_data/gt \
      --patch_dir  ./patches \
      --patch_size 256 --stride 256

  # 학습
  python skin_train_v2.py

  # 이어서 학습
  python skin_train_v2.py --resume ./checkpoints_v2/last_checkpoint.pth
"""

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

from skin_net_v2      import build_analyzer_v2
from skin_loss_smooth import SkinAnalyzerLoss, get_loss_weights
from skin_dataset     import SkinDataset, skin_collate_fn, ExcludedDataset, preview_collate_fn
from ambient_aug      import apply_batch_illumination_aug


# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    patch_dir      = './patches',
    checkpoint_dir = './checkpoints_v2',    # smooth/enhance와 분리

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
    dice_threshold = 0.5,
    best_w_brown   = 1.0,
    best_w_red     = 1.0,
    best_w_wrinkle = 1.2,

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
    preview_dir       = './previews_v2',
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
# freq_reg 계산 — V2 모델 전용
# ══════════════════════════════════════════════════════════════════════════════
def _v2_freq_reg(model: torch.nn.Module) -> torch.Tensor:
    """
    ODFrequencyFilter.low_logit 정규화 손실

    gate_low = sigmoid(low_logit) → 값이 클수록 조명 억제 강함
    w_freq_reg가 이 값을 높이도록 유도 → 조명 성분을 더 적극적으로 제거
    기존 _freq_reg(freq1~3 평균)와 동일한 역할, V2 구조에 맞게 재구현
    """
    gate = torch.sigmoid(model.cross_encoder.od_filter.low_logit)
    return gate.mean()


@torch.no_grad()
def _dice_sum_and_count(pred_logit: torch.Tensor,
                        gt:         torch.Tensor,
                        face_mask:  torch.Tensor,
                        has_gt:     list[bool],
                        threshold:  float = 0.5,
                        eps:        float = 1e-6) -> tuple[float, int]:
    """
    부분 GT 지원 Dice metric.

    - sigmoid + threshold 후 binary mask 기준으로 계산
    - GT가 존재하는 샘플(has_gt=True)만 평균
    - face_mask 내부 픽셀만 평가
    """
    valid = torch.tensor(has_gt, dtype=torch.bool, device=pred_logit.device)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return 0.0, 0

    pred_bin = (torch.sigmoid(pred_logit) >= threshold).float() * face_mask
    gt_bin   = (gt >= 0.5).float() * face_mask

    pred_flat = pred_bin[valid].flatten(1)
    gt_flat   = gt_bin[valid].flatten(1)

    inter = (pred_flat * gt_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
    dice  = (2.0 * inter + eps) / (denom + eps)
    return float(dice.sum().item()), n_valid


def _weighted_best_score(val_log: dict, cfg: dict) -> float:
    wb = cfg.get('best_w_brown', 1.0)
    wr = cfg.get('best_w_red', 1.0)
    ww = cfg.get('best_w_wrinkle', 1.0)
    denom = wb + wr + ww
    return (
        wb * val_log.get('brown_dice', 0.0) +
        wr * val_log.get('red_dice', 0.0) +
        ww * val_log.get('wrinkle_dice', 0.0)
    ) / max(denom, 1e-6)


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

        rgb_cross_aug    = apply_batch_illumination_aug(rgb_cross,    **aug_kwargs)
        rgb_parallel_aug = apply_batch_illumination_aug(rgb_parallel, **aug_kwargs)
        result = model(rgb_cross_aug, rgb_parallel_aug, mask)

        result_clean = None
        if criterion.w_consist > 0:
            with torch.no_grad():
                result_clean = model(rgb_cross, rgb_parallel, mask)

        # freq_reg는 V2 구조에 맞게 직접 계산 (model=None 전달)
        loss, detail = criterion(
            result,
            brown_gt, red_gt, wrinkle_gt,
            rgb_cross   = rgb_cross,
            face_mask   = mask,
            has_brown   = has_brown,
            has_red     = has_red,
            has_wrinkle = has_wrinkle,
            result_aug  = result_clean,
            model       = None,
        )

        if criterion.w_freq_reg > 0:
            l_freq = _v2_freq_reg(model)
            loss   = loss + criterion.w_freq_reg * l_freq
            detail['freq_reg'] = l_freq.item()

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
    dice_sum = dict(brown=0.0, red=0.0, wrinkle=0.0)
    dice_cnt = dict(brown=0, red=0, wrinkle=0)

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

        brown_sum, brown_cnt = _dice_sum_and_count(
            result.brown_mask, brown_gt, mask, batch['has_brown'],
            threshold=CFG.get('dice_threshold', 0.5),
        )
        red_sum, red_cnt = _dice_sum_and_count(
            result.red_mask, red_gt, mask, batch['has_red'],
            threshold=CFG.get('dice_threshold', 0.5),
        )
        wrinkle_sum, wrinkle_cnt = _dice_sum_and_count(
            result.wrinkle_mask, wrinkle_gt, mask, batch['has_wrinkle'],
            threshold=CFG.get('dice_threshold', 0.5),
        )

        dice_sum['brown'] += brown_sum
        dice_sum['red'] += red_sum
        dice_sum['wrinkle'] += wrinkle_sum
        dice_cnt['brown'] += brown_cnt
        dice_cnt['red'] += red_cnt
        dice_cnt['wrinkle'] += wrinkle_cnt

        pbar.set_postfix(loss=f"{total_loss / step:.4f}")

    pbar.close()
    n = max(len(loader), 1)
    out = {'total': total_loss / n, **{k: v / n for k, v in log.items()}}
    out['brown_dice'] = dice_sum['brown'] / max(dice_cnt['brown'], 1)
    out['red_dice'] = dice_sum['red'] / max(dice_cnt['red'], 1)
    out['wrinkle_dice'] = dice_sum['wrinkle'] / max(dice_cnt['wrinkle'], 1)
    out['score'] = _weighted_best_score(out, CFG)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='SkinAnalyzerV2 학습 (H1+H2 수정)')
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
    print(f"Model   : SkinAnalyzerV2")
    print(f"  H1: HueBranch + SpotHeadV2  (Brown/Red 색조 구분 강화)")
    print(f"  H2: CrossPolEncoderV2       (OD 공간 직접 주파수 분리, Beer-Lambert 준수)")
    print(f"Loss    : skin_loss_smooth    (Phase warm-up, early freq_reg=0.02)")

    ckpt_dir = Path(CFG['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_analyzer_v2(
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
        all_weights   = full_ds.get_sample_weights(
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
        w_brown   = CFG['w_brown'],
        w_red     = CFG['w_red'],
        w_wrinkle = CFG['w_wrinkle'],
        w_recon   = CFG['w_recon'],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_score  = float('-inf')

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
        best_score  = ckpt.get('best_score', ckpt.get('val_metric', {}).get('score', float('-inf')))
        print(f"Resume: epoch {ckpt['epoch']} → {start_epoch}부터 재개  "
              f"(best_score={best_score:.4f}  lr={scheduler.get_last_lr()[0]:.2e})")
    # ─────────────────────────────────────────────────────────────────────────

    print("\n학습 시작...")

    epoch_bar = tqdm(range(start_epoch, CFG['epochs'] + 1),
                     desc='Epochs', unit='ep', dynamic_ncols=True)

    for epoch in epoch_bar:
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

        # V2 gate 로깅: ODFrequencyFilter.low_logit (R/G/B 채널별)
        od_gate = torch.sigmoid(
            model.cross_encoder.od_filter.low_logit
        ).squeeze().tolist()   # [gate_R, gate_G, gate_B]

        is_best = val_log['score'] > best_score

        epoch_bar.set_postfix(
            train=f"{train_log['total']:.4f}",
            val=f"{val_log['total']:.4f}",
            bD=f"{val_log['brown_dice']:.3f}",
            rD=f"{val_log['red_dice']:.3f}",
            wD=f"{val_log['wrinkle_dice']:.3f}",
        )

        tqdm.write(
            f"[{epoch:03d}/{CFG['epochs']}]"
            f"  trn={train_log['total']:.4f}"
            f"  val={val_log['total']:.4f}"
            f"  B={val_log['brown']:.4f}"
            f"  R={val_log['red']:.4f}"
            f"  W={val_log['wrinkle']:.4f}"
            f"  B_dice={val_log['brown_dice']:.3f}"
            f"  R_dice={val_log['red_dice']:.3f}"
            f"  W_dice={val_log['wrinkle_dice']:.3f}"
            f"  score={val_log['score']:.3f}"
            f"  rc={val_log['recon']:.4f}"
            f"  od_gate(R,G,B)=[{od_gate[0]:.3f},{od_gate[1]:.3f},{od_gate[2]:.3f}]"
            f"  w_consist={criterion.w_consist:.3f}"
            f"  w_freq={criterion.w_freq_reg:.3f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
            + (' ★' if is_best else '')
        )

        if is_best:
            best_score = val_log['score']

        last_ckpt = {
            'epoch'                : epoch,
            'state_dict'           : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'best_score'           : best_score,
            'val_loss'             : val_log,
            'val_metric'           : {
                'brown_dice': val_log['brown_dice'],
                'red_dice': val_log['red_dice'],
                'wrinkle_dice': val_log['wrinkle_dice'],
                'score': val_log['score'],
            },
            'cfg'                  : CFG,
        }
        torch.save(last_ckpt, ckpt_dir / 'last_checkpoint.pth')

        if is_best:
            torch.save(last_ckpt, ckpt_dir / 'best_skin_analyzer.pth')
            tqdm.write(
                "  ★ Best 저장:"
                f" score={best_score:.3f}"
                f" (B={val_log['brown_dice']:.3f}"
                f", R={val_log['red_dice']:.3f}"
                f", W={val_log['wrinkle_dice']:.3f})"
            )

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
