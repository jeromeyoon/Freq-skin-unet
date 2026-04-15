"""
train.py
========
FreqAwareUNet 학습 스크립트

학습 전략
---------
Phase 1 (0~40%)  : supervised L1 + Beer-Lambert recon 만 → 빠른 수렴
Phase 2 (40~80%) : + illumination consistency (조명 augment 일관성)
Phase 3 (80~100%): + frequency gate regularization (low freq 억제 강화)

실행
----
  python train.py
  nohup python train.py > train.log 2>&1 &
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from model      import build_model
from loss       import MultiTaskSkinLoss, get_loss_weights
from dataset    import SkinDataset
from ambient_aug import apply_batch_illumination_aug


# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # 경로
    patch_dir      = './patches',           # SkinDataset 루트
    checkpoint_dir = './checkpoints',

    # 모델
    base_ch        = 64,                    # U-Net 기본 채널
    low_r          = 0.1,                   # 저주파(조명) 억제 반경
    high_r         = 0.4,                   # 고주파(주름) 시작 반경
    img_size       = 256,

    # 학습
    epochs         = 100,
    batch_size     = 16,
    lr             = 1e-4,
    weight_decay   = 1e-4,
    num_workers    = 4,
    val_ratio      = 0.1,                   # 전체 데이터 중 검증 비율

    # Loss 초기 가중치 (get_loss_weights로 단계적 조절)
    w_brown        = 1.0,
    w_red          = 1.0,
    w_wrinkle      = 1.0,
    w_recon        = 0.3,
    w_consist      = 0.0,
    w_freq_reg     = 0.0,

    # Illumination augmentation 설정
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

    for batch in loader:
        rgb       = batch['rgb'].to(device)
        brown_gt  = batch['brown'].to(device)
        red_gt    = batch['red'].to(device)
        wrinkle_gt= batch['wrinkle'].to(device)
        mask      = batch['mask'].to(device)

        # 원본 예측
        brown, red, wrinkle = model(rgb)

        # 조명 변형 예측 (consistency loss가 활성화된 경우만)
        brown_aug = red_aug = wrinkle_aug = None
        if criterion.w_consist > 0:
            rgb_aug = apply_batch_illumination_aug(
                rgb,
                intensity_range  = cfg['intensity_range'],
                color_temp_range = cfg['color_temp_range'],
                tint_range       = cfg['tint_range'],
                vignette_prob    = cfg['vignette_prob'],
            )
            with torch.no_grad():
                brown_aug, red_aug, wrinkle_aug = model(rgb_aug)

        loss, detail = criterion(
            brown, red, wrinkle,
            brown_gt, red_gt, wrinkle_gt,
            rgb, mask,
            brown_aug, red_aug, wrinkle_aug,
            model if criterion.w_freq_reg > 0 else None,
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
def validate(model, loader, criterion, device, cfg):
    model.eval()

    total_loss = 0.0
    log = {k: 0.0 for k in ['brown', 'red', 'wrinkle', 'recon']}

    for batch in loader:
        rgb        = batch['rgb'].to(device)
        brown_gt   = batch['brown'].to(device)
        red_gt     = batch['red'].to(device)
        wrinkle_gt = batch['wrinkle'].to(device)
        mask       = batch['mask'].to(device)

        brown, red, wrinkle = model(rgb)

        loss, detail = criterion(
            brown, red, wrinkle,
            brown_gt, red_gt, wrinkle_gt,
            rgb, mask,
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

    # ── 모델 ──────────────────────────────────────────────────────────────────
    model = build_model(
        base_ch = CFG['base_ch'],
        low_r   = CFG['low_r'],
        high_r  = CFG['high_r'],
    ).to(device)

    # ── Dataset ────────────────────────────────────────────────────────────────
    full_ds = SkinDataset(
        patch_dir = CFG['patch_dir'],
        img_size  = CFG['img_size'],
        augment   = True,
    )
    val_size   = max(1, int(len(full_ds) * CFG['val_ratio']))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # val에서 augment 끄기
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds, batch_size=CFG['batch_size'],
        shuffle=True, num_workers=CFG['num_workers'], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG['batch_size'],
        shuffle=False, num_workers=CFG['num_workers'], pin_memory=True,
    )
    print(f"Train: {len(train_ds)} / Val: {len(val_ds)}")

    # ── Loss ───────────────────────────────────────────────────────────────────
    criterion = MultiTaskSkinLoss(
        w_brown    = CFG['w_brown'],
        w_red      = CFG['w_red'],
        w_wrinkle  = CFG['w_wrinkle'],
        w_recon    = CFG['w_recon'],
        w_consist  = CFG['w_consist'],
        w_freq_reg = CFG['w_freq_reg'],
    ).to(device)

    # ── Optimizer / Scheduler ──────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs']
    )

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    print("\n학습 시작...")
    best_val = float('inf')

    for epoch in range(1, CFG['epochs'] + 1):

        # 단계별 loss 가중치 업데이트
        weights = get_loss_weights(epoch, CFG['epochs'])
        for k, v in weights.items():
            setattr(criterion, k, v)

        train_log = train_one_epoch(model, train_loader, criterion, optimizer, device, CFG)
        val_log   = validate(model, val_loader, criterion, device, CFG)
        scheduler.step()

        # 현재 FrequencyGate low_gate 값 확인 (조명 억제 정도)
        low_gates = [
            torch.sigmoid(model.freq1.low_logit).mean().item(),
            torch.sigmoid(model.freq2.low_logit).mean().item(),
            torch.sigmoid(model.freq3.low_logit).mean().item(),
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
            ckpt_path = Path(CFG['checkpoint_dir']) / 'best_model.pth'
            torch.save({
                'epoch'     : epoch,
                'state_dict': model.state_dict(),
                'val_loss'  : val_log,
                'cfg'       : CFG,
            }, ckpt_path)
            print(f"  ✅ Best 저장: val={best_val:.4f}")

    print("\n학습 완료!")


if __name__ == '__main__':
    main()
