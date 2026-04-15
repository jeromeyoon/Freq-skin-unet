"""
skin_train.py
=============
SkinAnalyzer 듀얼 편광 + 부분 GT 학습 스크립트

실행
----
  # 패치 먼저 준비
  python data_prep.py \
      --input_path ./raw_data/images \
      --gt_path    ./raw_data/gt \
      --patch_dir  ./patches \
      --patch_size 256 --stride 256

  # 학습
  python skin_train.py
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

import torchvision.utils as vutils
import torchvision.transforms.functional as TF

from skin_net     import build_analyzer
from skin_loss    import SkinAnalyzerLoss, get_loss_weights
from skin_dataset import SkinDataset, skin_collate_fn, ExcludedDataset, preview_collate_fn
from ambient_aug  import apply_batch_illumination_aug


# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    patch_dir      = './patches',
    checkpoint_dir = './checkpoints',

    base_ch        = 64,
    low_r          = 0.1,
    high_r         = 0.4,
    img_size       = 256,

    epochs         = 100,
    batch_size     = 16,
    lr             = 1e-4,
    weight_decay   = 1e-4,
    num_workers    = 4,
    val_ratio      = 0.1,

    w_brown        = 1.0,
    w_red          = 1.0,
    w_wrinkle      = 1.0,
    w_recon        = 0.3,
    w_consist      = 0.0,
    w_freq_reg     = 0.0,

    intensity_range   = (0.5, 1.8),
    color_temp_range  = (0.7, 1.3),
    tint_range        = (0.8, 1.2),
    vignette_prob     = 0.3,

    # 학습 제외 이미지 시각화
    preview_interval  = 10,    # N 에폭마다 저장 (0이면 비활성)
    preview_n_images  = 8,     # 저장할 이미지 수
    preview_dir       = './previews',
)


# ══════════════════════════════════════════════════════════════════════════════
# 학습 제외 이미지 예측 결과 시각화
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def save_preview(model, preview_loader, device, epoch, save_dir: Path, n_images: int):
    """
    GT 없는(학습 제외) 이미지에 대한 예측 결과를 이미지로 저장.

    저장 형식: {save_dir}/epoch_{epoch:03d}/{stem}.png
    각 이미지: [rgb_cross | brown_pred | red_pred | wrinkle_pred] 가로 나열
    """
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

            # 각 채널을 0~1 범위로 클램프
            inp   = rgb_cross[i].cpu().clamp(0, 1)           # [3,H,W]
            brown = result.brown_mask[i].cpu().clamp(0, 1)   # [1,H,W]
            red   = result.red_mask[i].cpu().clamp(0, 1)     # [1,H,W]
            wrk   = result.wrinkle_mask[i].cpu().clamp(0, 1) # [1,H,W]

            # 단채널 → RGB 반복
            brown_rgb = brown.repeat(3, 1, 1)
            red_rgb   = red.repeat(3, 1, 1)
            wrk_rgb   = wrk.repeat(3, 1, 1)

            # 가로 격자: [입력 | brown | red | wrinkle]
            grid = vutils.make_grid(
                torch.stack([inp, brown_rgb, red_rgb, wrk_rgb]),
                nrow=4, padding=2, normalize=False,
            )
            img = TF.to_pil_image(grid)
            img.save(save_dir / f'{stems[i]}.png')
            saved += 1

    if saved:
        print(f"  Preview 저장: {save_dir}  ({saved}장)")


# ══════════════════════════════════════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device, cfg):
    model.train()
    total_loss = 0.0
    log = {k: 0.0 for k in ['brown', 'red', 'wrinkle', 'recon', 'consist', 'freq_reg']}

    aug_kwargs = dict(
        intensity_range  = cfg['intensity_range'],
        color_temp_range = cfg['color_temp_range'],
        tint_range       = cfg['tint_range'],
        vignette_prob    = cfg['vignette_prob'],
    )

    for batch in loader:
        rgb_cross    = batch['rgb_cross'].to(device)
        rgb_parallel = batch['rgb_parallel'].to(device)
        brown_gt     = batch['brown'].to(device)
        red_gt       = batch['red'].to(device)
        wrinkle_gt   = batch['wrinkle'].to(device)
        mask         = batch['mask'].to(device)

        has_brown   = batch['has_brown']    # list[bool]
        has_red     = batch['has_red']
        has_wrinkle = batch['has_wrinkle']

        result = model(rgb_cross, rgb_parallel, mask)

        # Consistency augmentation
        result_aug = None
        if criterion.w_consist > 0:
            rgb_cross_aug    = apply_batch_illumination_aug(rgb_cross,    **aug_kwargs)
            rgb_parallel_aug = apply_batch_illumination_aug(rgb_parallel, **aug_kwargs)
            with torch.no_grad():
                result_aug = model(rgb_cross_aug, rgb_parallel_aug, mask)

        loss, detail = criterion(
            result,
            brown_gt, red_gt, wrinkle_gt,
            rgb_cross   = rgb_cross,
            face_mask   = mask,
            has_brown   = has_brown,
            has_red     = has_red,
            has_wrinkle = has_wrinkle,
            result_aug  = result_aug,
            model       = model if criterion.w_freq_reg > 0 else None,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in log:
            log[k] += detail.get(k, 0.0)

    n = len(loader)
    return {'total': total_loss / n, **{k: v / n for k, v in log.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# 검증
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    log = {k: 0.0 for k in ['brown', 'red', 'wrinkle', 'recon']}

    for batch in loader:
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

    n = len(loader)
    return {'total': total_loss / n, **{k: v / n for k, v in log.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    Path(CFG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    model = build_analyzer(
        base_ch = CFG['base_ch'],
        low_r   = CFG['low_r'],
        high_r  = CFG['high_r'],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset + DataLoader (skin_collate_fn 사용)
    full_ds  = SkinDataset(CFG['patch_dir'], CFG['img_size'], augment=True)
    val_size = max(1, int(len(full_ds) * CFG['val_ratio']))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds, CFG['batch_size'], shuffle=True,
        num_workers=CFG['num_workers'], pin_memory=True,
        collate_fn=skin_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, CFG['batch_size'], shuffle=False,
        num_workers=CFG['num_workers'], pin_memory=True,
        collate_fn=skin_collate_fn,
    )
    print(f"Train: {len(train_ds)} / Val: {len(val_ds)}")

    # 학습 제외 이미지 (GT 없음) → 시각화 전용 loader
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
            print(f"Preview 대상: {len(excl_ds)}장 "
                  f"(매 {CFG['preview_interval']} 에폭마다 저장)")
        else:
            print("Preview 대상 없음 (GT 없는 패치가 존재하지 않음)")

    criterion = SkinAnalyzerLoss(
        w_brown    = CFG['w_brown'],
        w_red      = CFG['w_red'],
        w_wrinkle  = CFG['w_wrinkle'],
        w_recon    = CFG['w_recon'],
        w_consist  = CFG['w_consist'],
        w_freq_reg = CFG['w_freq_reg'],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])

    print("\n학습 시작...")
    best_val = float('inf')

    for epoch in range(1, CFG['epochs'] + 1):
        weights = get_loss_weights(epoch, CFG['epochs'])
        for k, v in weights.items():
            setattr(criterion, k, v)

        train_log = train_one_epoch(model, train_loader, criterion, optimizer, device, CFG)
        val_log   = validate(model, val_loader, criterion, device)
        scheduler.step()

        enc = model.cross_encoder
        low_gates = [
            torch.sigmoid(enc.freq1.low_logit).mean().item(),
            torch.sigmoid(enc.freq2.low_logit).mean().item(),
            torch.sigmoid(enc.freq3.low_logit).mean().item(),
        ]

        print(
            f"Epoch [{epoch:03d}/{CFG['epochs']}]"
            f"  train={train_log['total']:.4f}"
            f"  val={val_log['total']:.4f}"
            f"  brown={val_log['brown']:.4f}"
            f"  red={val_log['red']:.4f}"
            f"  wrinkle={val_log['wrinkle']:.4f}"
            f"  recon={val_log['recon']:.4f}"
            f"  low_gate=[{low_gates[0]:.3f},{low_gates[1]:.3f},{low_gates[2]:.3f}]"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_log['total'] < best_val:
            best_val  = val_log['total']
            torch.save({
                'epoch'     : epoch,
                'state_dict': model.state_dict(),
                'val_loss'  : val_log,
                'cfg'       : CFG,
            }, Path(CFG['checkpoint_dir']) / 'best_skin_analyzer.pth')
            print(f"  Best 저장: val={best_val:.4f}")

        # 학습 제외 이미지 예측 결과 시각화
        if (preview_loader is not None
                and CFG['preview_interval'] > 0
                and epoch % CFG['preview_interval'] == 0):
            save_preview(
                model, preview_loader, device,
                epoch,
                save_dir  = Path(CFG['preview_dir']),
                n_images  = CFG['preview_n_images'],
            )

    print("\n학습 완료!")


if __name__ == '__main__':
    main()
