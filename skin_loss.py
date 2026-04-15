"""
skin_loss.py
============
SkinAnalyzerLoss : 듀얼 편광 입력 학습 Loss

Beer-Lambert recon은 교차 편광(rgb_cross)만 사용
  → 교차 편광이 발색단 흡수 물리 모델에 적합

학습 단계
---------
Phase 1 (0~40%) : supervised L1 + Beer-Lambert recon (교차 편광)
Phase 2 (40~80%): + illumination consistency
Phase 3 (80~100%): + freq gate regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from skin_net import SkinResult


# ══════════════════════════════════════════════════════════════════════════════
# SkinAnalyzerLoss
# ══════════════════════════════════════════════════════════════════════════════
class SkinAnalyzerLoss(nn.Module):
    """
    Parameters
    ----------
    w_brown    : 갈색 반점 supervised loss 가중치
    w_red      : 적색 반점 supervised loss 가중치
    w_wrinkle  : 주름 supervised loss 가중치
    w_recon    : Beer-Lambert 재구성 loss 가중치 (교차 편광 기준)
    w_consist  : 조명 일관성 loss 가중치
    w_freq_reg : 주파수 게이트 정규화 가중치
    """

    _MEL_ABS = [0.28, 0.18, 0.09]   # 멜라닌 채널별 흡수 계수 (R,G,B)
    _HEM_ABS = [0.10, 0.35, 0.05]   # 헤모글로빈 채널별 흡수 계수

    def __init__(self,
                 w_brown:    float = 1.0,
                 w_red:      float = 1.0,
                 w_wrinkle:  float = 1.0,
                 w_recon:    float = 0.3,
                 w_consist:  float = 0.0,
                 w_freq_reg: float = 0.0):
        super().__init__()
        self.w_brown    = w_brown
        self.w_red      = w_red
        self.w_wrinkle  = w_wrinkle
        self.w_recon    = w_recon
        self.w_consist  = w_consist
        self.w_freq_reg = w_freq_reg

        self.register_buffer('mel_abs',
                             torch.tensor(self._MEL_ABS).view(1, 3, 1, 1))
        self.register_buffer('hem_abs',
                             torch.tensor(self._HEM_ABS).view(1, 3, 1, 1))

    def _masked_l1(self, pred, gt, mask):
        denom = mask.sum().clamp(min=1.0)
        return (F.l1_loss(pred, gt, reduction='none') * mask).sum() / denom

    def _beer_lambert_recon(self, brown_mask, red_mask, rgb_cross, mask):
        """교차 편광 RGB를 발색단으로 재구성 후 L1 비교"""
        od_recon  = brown_mask * self.mel_abs + red_mask * self.hem_abs
        rgb_recon = torch.exp(-od_recon).clamp(0.0, 1.0)
        return self._masked_l1(rgb_recon, rgb_cross, mask)

    @staticmethod
    def _freq_reg(model: nn.Module) -> torch.Tensor:
        """교차 편광 인코더의 FrequencyGate low_logit을 0으로 억제"""
        enc = model.cross_encoder
        gates = [
            torch.sigmoid(enc.freq1.low_logit).mean(),
            torch.sigmoid(enc.freq2.low_logit).mean(),
            torch.sigmoid(enc.freq3.low_logit).mean(),
        ]
        return sum(gates) / len(gates)

    def forward(self,
                result:       SkinResult,
                brown_gt:     torch.Tensor,
                red_gt:       torch.Tensor,
                wrinkle_gt:   torch.Tensor,
                rgb_cross:    torch.Tensor,   # Beer-Lambert recon용 교차 편광
                face_mask:    torch.Tensor,
                result_aug:   SkinResult | None = None,
                model:        nn.Module  | None = None):
        """
        Parameters
        ----------
        result      : SkinResult (model forward 출력)
        brown_gt    : [B,1,H,W]
        red_gt      : [B,1,H,W]
        wrinkle_gt  : [B,1,H,W]
        rgb_cross   : [B,3,H,W]  교차 편광 원본 (recon loss용)
        face_mask   : [B,1,H,W]
        result_aug  : SkinResult (조명 aug 후, consistency 비활성 시 None)
        model       : SkinAnalyzer (freq_reg 비활성 시 None)

        Returns
        -------
        total_loss : scalar tensor
        detail     : dict
        """
        detail = {}

        # ── Supervised L1 ────────────────────────────────────────────────────
        l_brown   = self._masked_l1(result.brown_mask,   brown_gt,   face_mask)
        l_red     = self._masked_l1(result.red_mask,     red_gt,     face_mask)
        l_wrinkle = self._masked_l1(result.wrinkle_mask, wrinkle_gt, face_mask)

        detail.update(brown=l_brown.item(), red=l_red.item(), wrinkle=l_wrinkle.item())

        loss = (self.w_brown   * l_brown +
                self.w_red     * l_red   +
                self.w_wrinkle * l_wrinkle)

        # ── Beer-Lambert recon (교차 편광) ───────────────────────────────────
        l_recon = self._beer_lambert_recon(
            result.brown_mask, result.red_mask, rgb_cross, face_mask)
        detail['recon'] = l_recon.item()
        loss = loss + self.w_recon * l_recon

        # ── Consistency ──────────────────────────────────────────────────────
        if self.w_consist > 0 and result_aug is not None:
            l_consist = (
                self._masked_l1(result_aug.brown_mask,   result.brown_mask.detach(),   face_mask) +
                self._masked_l1(result_aug.red_mask,     result.red_mask.detach(),     face_mask) +
                self._masked_l1(result_aug.wrinkle_mask, result.wrinkle_mask.detach(), face_mask)
            ) / 3.0
            detail['consist'] = l_consist.item()
            loss = loss + self.w_consist * l_consist
        else:
            detail['consist'] = 0.0

        # ── Freq regularization ──────────────────────────────────────────────
        if self.w_freq_reg > 0 and model is not None:
            l_freq = self._freq_reg(model)
            detail['freq_reg'] = l_freq.item()
            loss = loss + self.w_freq_reg * l_freq
        else:
            detail['freq_reg'] = 0.0

        return loss, detail


# ══════════════════════════════════════════════════════════════════════════════
# 단계별 Loss 가중치 스케줄
# ══════════════════════════════════════════════════════════════════════════════
def get_loss_weights(epoch: int, total_epochs: int) -> dict:
    progress = epoch / total_epochs
    base = dict(w_brown=1.0, w_red=1.0, w_wrinkle=1.0,
                w_recon=0.3, w_consist=0.0, w_freq_reg=0.0)

    if progress < 0.4:
        return base
    if progress < 0.8:
        return {**base, 'w_consist': 0.3}
    return {**base, 'w_consist': 0.3, 'w_freq_reg': 0.1}
