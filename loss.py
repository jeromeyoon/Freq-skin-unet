"""
loss.py
=======
MultiTaskSkinLoss : FreqAwareUNet 학습 Loss

구성
----
1. Supervised L1      : VISIA GT와 직접 비교 (brown, red, wrinkle)
2. Beer-Lambert Recon : OD → RGB 복원 물리 제약 (brown/red만)
3. Freq Consistency   : 조명 변형 이미지 → 같은 출력 (illumination augment)
4. Frequency Reg      : low_gate를 0으로 유도 (조명 억제 강화)

단계적 weight 스케줄:
  Phase 1 (0~40%) : supervised + recon 만
  Phase 2 (40~80%): + freq_consistency
  Phase 3 (80~100%): + freq_reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskSkinLoss(nn.Module):
    """
    Parameters
    ----------
    w_brown      : melanin L1 가중치
    w_red        : hemoglobin L1 가중치
    w_wrinkle    : wrinkle L1 가중치
    w_recon      : Beer-Lambert 재구성 가중치
    w_consist    : 조명 일관성 가중치
    w_freq_reg   : FrequencyGate low_gate 억제 정규화
    """

    # Beer-Lambert 흡수 계수 [R, G, B]
    _MEL_ABS = [0.28, 0.18, 0.09]
    _HEM_ABS = [0.10, 0.35, 0.05]

    def __init__(self,
                 w_brown   : float = 1.0,
                 w_red     : float = 1.0,
                 w_wrinkle : float = 1.0,
                 w_recon   : float = 0.3,
                 w_consist : float = 0.0,
                 w_freq_reg: float = 0.0):
        super().__init__()

        self.w_brown    = w_brown
        self.w_red      = w_red
        self.w_wrinkle  = w_wrinkle
        self.w_recon    = w_recon
        self.w_consist  = w_consist
        self.w_freq_reg = w_freq_reg

        mel_abs = torch.tensor(self._MEL_ABS).view(1, 3, 1, 1)
        hem_abs = torch.tensor(self._HEM_ABS).view(1, 3, 1, 1)
        self.register_buffer('mel_abs', mel_abs)
        self.register_buffer('hem_abs', hem_abs)

    @staticmethod
    def _masked_l1(pred: torch.Tensor,
                   gt  : torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        return (torch.abs(pred - gt) * mask).sum() / (mask.sum() + 1e-6)

    def _beer_lambert_recon(self,
                             brown     : torch.Tensor,
                             red       : torch.Tensor,
                             rgb_linear: torch.Tensor,
                             mask      : torch.Tensor) -> torch.Tensor:
        """
        OD_recon = brown * mel_abs + red * hem_abs
        RGB_recon = exp(-OD_recon)
        → L1(RGB_recon, rgb_linear) 마스크 영역 내
        """
        od_recon  = brown * self.mel_abs + red * self.hem_abs
        rgb_recon = torch.exp(-od_recon)
        mask_3ch  = mask.expand_as(rgb_recon)
        return (torch.abs(rgb_recon - rgb_linear) * mask_3ch).sum() / \
               (mask_3ch.sum() + 1e-6)

    @staticmethod
    def _freq_reg(model: nn.Module) -> torch.Tensor:
        """
        FrequencyGate.low_logit 를 음수로 유도 → sigmoid → 0 (조명 억제)
        loss = mean(sigmoid(low_logit))  → 0 으로 학습
        """
        total = torch.tensor(0.0)
        for name, module in model.named_modules():
            if hasattr(module, 'low_logit'):
                gate = torch.sigmoid(module.low_logit)
                total = total + gate.mean()
        return total

    def forward(self,
                # 원본 예측
                brown      : torch.Tensor,   # [B, 1, H, W]
                red        : torch.Tensor,   # [B, 1, H, W]
                wrinkle    : torch.Tensor,   # [B, 1, H, W]
                # VISIA GT
                brown_gt   : torch.Tensor,
                red_gt     : torch.Tensor,
                wrinkle_gt : torch.Tensor,
                # 입력
                rgb_linear : torch.Tensor,   # [B, 3, H, W]
                face_mask  : torch.Tensor,   # [B, 1, H, W]
                # 조명 변형 예측 (선택)
                brown_aug  : torch.Tensor = None,
                red_aug    : torch.Tensor = None,
                wrinkle_aug: torch.Tensor = None,
                # freq reg용 model (선택)
                model      : nn.Module    = None,
                ) -> tuple:

        # ── 1. Supervised L1 ─────────────────────────────────────────────────
        l_brown   = self._masked_l1(brown,   brown_gt,   face_mask)
        l_red     = self._masked_l1(red,     red_gt,     face_mask)
        l_wrinkle = self._masked_l1(wrinkle, wrinkle_gt, face_mask)

        # ── 2. Beer-Lambert Reconstruction ───────────────────────────────────
        l_recon = self._beer_lambert_recon(brown, red, rgb_linear, face_mask)

        total = (self.w_brown   * l_brown
               + self.w_red     * l_red
               + self.w_wrinkle * l_wrinkle
               + self.w_recon   * l_recon)

        detail = {
            'brown'   : l_brown.item(),
            'red'     : l_red.item(),
            'wrinkle' : l_wrinkle.item(),
            'recon'   : l_recon.item(),
            'consist' : 0.0,
            'freq_reg': 0.0,
        }

        # ── 3. Illumination Consistency ───────────────────────────────────────
        if self.w_consist > 0 and brown_aug is not None:
            l_consist = (F.l1_loss(brown,   brown_aug)
                       + F.l1_loss(red,     red_aug)
                       + F.l1_loss(wrinkle, wrinkle_aug))
            total              += self.w_consist * l_consist
            detail['consist']   = l_consist.item()

        # ── 4. Frequency Gate Regularization ─────────────────────────────────
        if self.w_freq_reg > 0 and model is not None:
            l_freq_reg = self._freq_reg(model)
            total              += self.w_freq_reg * l_freq_reg
            detail['freq_reg']  = l_freq_reg.item()

        detail['total'] = total.item()
        return total, detail


# ══════════════════════════════════════════════════════════════════════════════
# 학습 단계별 가중치 스케줄
# ══════════════════════════════════════════════════════════════════════════════
def get_loss_weights(epoch: int, total_epochs: int) -> dict:
    """
    Phase 1 (0~40%)  : supervised + recon → 빠른 수렴
    Phase 2 (40~80%) : + illumination consistency
    Phase 3 (80~100%): + frequency gate regularization

    Usage
    -----
    weights = get_loss_weights(epoch, total_epochs)
    for k, v in weights.items():
        setattr(criterion, k, v)
    """
    p = epoch / total_epochs

    if p < 0.4:
        return dict(w_brown=1.0, w_red=1.0, w_wrinkle=1.0,
                    w_recon=0.3, w_consist=0.0, w_freq_reg=0.0)
    elif p < 0.8:
        return dict(w_brown=1.0, w_red=1.0, w_wrinkle=1.0,
                    w_recon=0.3, w_consist=0.3, w_freq_reg=0.0)
    else:
        return dict(w_brown=1.0, w_red=1.0, w_wrinkle=1.0,
                    w_recon=0.3, w_consist=0.3, w_freq_reg=0.1)
