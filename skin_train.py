"""
skin_train.py
=============
SkinAnalyzer 듀얼 편광 학습 스크립트

입력
----
  rgb_cross    : 교차 편광 → 갈색/적색 반점 (발색단)
  rgb_parallel : 평행 편광 → 주름

Illumination Augmentation
--------------------------
  두 편광 이미지에 동일한 조명 파라미터 적용
  (동일 기기에서 촬영 → 같은 조명 환경 시뮬레이션)

실행
----
  python skin_train.py
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from skin_net     import build_analyzer
from skin_loss    import SkinAnalyzerLoss, get_loss_weights
from skin_dataset import SkinDataset
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
)


# ══════════════════════════════════════════════════════════════════════════════
# 학습 / 검증
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

        result = model(rgb_cross, rgb_parallel, mask)

        # Consistency: 동일 조명 파라미터로 두 편광 모두 aug
        result_aug = None
        if criterion.w_consist > 0:
            # 동일 seed로 두 이미지에 같은 조명 변형 적용
            rgb_cross_aug    = apply_batch_illumination_aug(rgb_cross,    **aug_kwargs)
            rgb_parallel_aug = apply_batch_illumination_aug(rgb_parallel, **aug_kwargs)
            with torch.no_grad():
                result_aug = model(rgb_cross_aug, rgb_parallel_aug, mask)

        loss, detail = criterion(
            result, brown_gt, red_gt, wrinkle_gt,
            rgb_cross  = rgb_cross,
            face_mask  = mask,
            result_aug = result_aug,
            model      = model if criterion.w_freq_reg > 0 else None,
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
            result, brown_gt, red_gt, wrinkle_gt,
            rgb_cross = rgb_cross,
            face_mask = mask,
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    full_ds  = SkinDataset(CFG['patch_dir'], CFG['img_size'], augment=True)
    val_size = max(1, int(len(full_ds) * CFG['val_ratio']))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, CFG['batch_size'],
                              shuffle=True,  num_workers=CFG['num_workers'], pin_memory=True)
    val_loader   = DataLoader(val_ds,   CFG['batch_size'],
                              shuffle=False, num_workers=CFG['num_workers'], pin_memory=True)
    print(f"Train: {len(train_ds)} / Val: {len(val_ds)}")

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

        # 교차 편광 인코더의 저주파 억제 모니터링
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
            ckpt_path = Path(CFG['checkpoint_dir']) / 'best_skin_analyzer.pth'
            torch.save({
                'epoch'     : epoch,
                'state_dict': model.state_dict(),
                'val_loss'  : val_log,
                'cfg'       : CFG,
            }, ckpt_path)
            print(f"  Best 저장: val={best_val:.4f}")

    print("\n학습 완료!")


if __name__ == '__main__':
    main()
