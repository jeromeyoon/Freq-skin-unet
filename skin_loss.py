"""
skin_loss.py
============
SkinAnalyzerLoss : 부분 GT(Partial Label) 지원 Loss

GT가 없는 샘플은 해당 task loss를 0으로 처리.
배치 내 일부 샘플만 GT가 있어도 올바르게 평균 계산.

Beer-Lambert recon은 교차 편광(rgb_cross)만 사용.

학습 단계
---------
Phase 1 (0~40%) : supervised L1 + Beer-Lambert recon
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

    _MEL_ABS = [0.28, 0.18, 0.09]
    _HEM_ABS = [0.10, 0.35, 0.05]

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

    # ── 부분 GT L1 ──────────────────────────────────────────────────────────
    def _partial_l1(self,
                    pred:     torch.Tensor,
                    gt:       torch.Tensor,
                    face_mask:torch.Tensor,
                    has_gt:   list[bool]) -> torch.Tensor:
        """
        배치 내 GT가 있는 샘플만 loss 계산.

        pred      : [B,1,H,W]
        gt        : [B,1,H,W]  (GT 없는 샘플은 zeros)
        face_mask : [B,1,H,W]
        has_gt    : list[bool]  길이 B
        """
        # GT 가용 마스크 [B,1,1,1]
        gt_avail = torch.tensor(has_gt, dtype=torch.float32,
                                device=pred.device).view(-1, 1, 1, 1)

        # 픽셀 단위 L1 × face_mask × GT 가용 여부
        per_pixel = F.l1_loss(pred, gt, reduction='none') * face_mask * gt_avail
        denom = (face_mask * gt_avail).sum().clamp(min=1.0)
        return per_pixel.sum() / denom

    # ── Beer-Lambert recon ──────────────────────────────────────────────────
    def _beer_lambert_recon(self,
                            brown_mask: torch.Tensor,
                            red_mask:   torch.Tensor,
                            rgb_cross:  torch.Tensor,
                            face_mask:  torch.Tensor,
                            has_brown:  list[bool],
                            has_red:    list[bool]) -> torch.Tensor:
        """
        brown 또는 red GT가 있는 샘플만 recon loss 계산.
        """
        has_any = [b or r for b, r in zip(has_brown, has_red)]
        if not any(has_any):
            return torch.tensor(0.0, device=rgb_cross.device)

        od_recon  = brown_mask * self.mel_abs + red_mask * self.hem_abs
        rgb_recon = torch.exp(-od_recon).clamp(0.0, 1.0)

        avail = torch.tensor(has_any, dtype=torch.float32,
                             device=rgb_cross.device).view(-1, 1, 1, 1)
        per_pixel = F.l1_loss(rgb_recon, rgb_cross, reduction='none') * face_mask * avail
        denom = (face_mask * avail).sum().clamp(min=1.0)
        return per_pixel.sum() / denom

    # ── 주파수 게이트 정규화 ─────────────────────────────────────────────────
    @staticmethod
    def _freq_reg(model: nn.Module) -> torch.Tensor:
        enc = model.cross_encoder
        gates = [
            torch.sigmoid(enc.freq1.low_logit).mean(),
            torch.sigmoid(enc.freq2.low_logit).mean(),
            torch.sigmoid(enc.freq3.low_logit).mean(),
        ]
        return sum(gates) / len(gates)

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self,
                result:      SkinResult,
                brown_gt:    torch.Tensor,
                red_gt:      torch.Tensor,
                wrinkle_gt:  torch.Tensor,
                rgb_cross:   torch.Tensor,
                face_mask:   torch.Tensor,
                has_brown:   list[bool],
                has_red:     list[bool],
                has_wrinkle: list[bool],
                result_aug:  SkinResult | None = None,
                model:       nn.Module  | None = None):
        """
        Parameters
        ----------
        has_brown / has_red / has_wrinkle : list[bool], 길이=B
            배치 내 샘플별 GT 가용 여부 (data_prep.py manifest 기반)
        """
        detail = {}

        # ── Supervised L1 (부분) ─────────────────────────────────────────────
        l_brown = self._partial_l1(
            result.brown_mask,   brown_gt,   face_mask, has_brown)
        l_red = self._partial_l1(
            result.red_mask,     red_gt,     face_mask, has_red)
        l_wrinkle = self._partial_l1(
            result.wrinkle_mask, wrinkle_gt, face_mask, has_wrinkle)

        detail.update(brown=l_brown.item(), red=l_red.item(), wrinkle=l_wrinkle.item())

        # 유효한 task만 loss에 합산 (모두 0인 경우 방지)
        n_active = sum([any(has_brown), any(has_red), any(has_wrinkle)])
        loss = torch.tensor(0.0, device=rgb_cross.device)
        if any(has_brown):
            loss = loss + self.w_brown   * l_brown
        if any(has_red):
            loss = loss + self.w_red     * l_red
        if any(has_wrinkle):
            loss = loss + self.w_wrinkle * l_wrinkle

        # ── Beer-Lambert recon ───────────────────────────────────────────────
        l_recon = self._beer_lambert_recon(
            result.brown_mask, result.red_mask, rgb_cross, face_mask,
            has_brown, has_red)
        detail['recon'] = l_recon.item()
        loss = loss + self.w_recon * l_recon

        # ── Consistency ──────────────────────────────────────────────────────
        if self.w_consist > 0 and result_aug is not None:
            l_consist = (
                self._partial_l1(result_aug.brown_mask,
                                 result.brown_mask.detach(), face_mask, has_brown) +
                self._partial_l1(result_aug.red_mask,
                                 result.red_mask.detach(),   face_mask, has_red)   +
                self._partial_l1(result_aug.wrinkle_mask,
                                 result.wrinkle_mask.detach(),face_mask, has_wrinkle)
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
