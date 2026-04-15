"""
skin_loss.py
============
SkinAnalyzerLoss : 부분 GT(Partial Label) 지원 Loss

GT가 없는 샘플은 해당 task loss를 0으로 처리.
배치 내 일부 샘플만 GT가 있어도 올바르게 평균 계산.

Beer-Lambert recon은 교차 편광(rgb_cross)만 사용.

Task별 Loss
-----------
  brown / red : BCE + Dice
    - BCE  : 픽셀별 확률 오차 (불균형 처리에 적합)
    - Dice : 영역 겹침 최대화 (희소 병변에 강인)
  wrinkle     : Focal + Dice
    - Focal: 어려운 예측(얇은 선)에 가중치 집중 (gamma=2, alpha=0.25)
    - Dice : 미세 구조 겹침 최대화

학습 단계
---------
Phase 1 (0~40%) : supervised (BCE/Focal+Dice) + Beer-Lambert recon
Phase 2 (40~80%): + illumination consistency
Phase 3 (80~100%): + freq gate regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from skin_net import SkinResult


# ══════════════════════════════════════════════════════════════════════════════
# 기본 Loss 함수
# ══════════════════════════════════════════════════════════════════════════════
def _dice_loss_per_sample(pred:      torch.Tensor,
                          gt:        torch.Tensor,
                          face_mask: torch.Tensor,
                          smooth:    float = 1.0) -> torch.Tensor:
    """
    샘플별 Dice Loss 반환.  [B] 크기 텐서.

    pred      : [B, 1, H, W]  logit (sigmoid 적용 전)
    gt        : [B, 1, H, W]  확률 GT
    face_mask : [B, 1, H, W]
    """
    prob = torch.sigmoid(pred)         # logit → prob
    # 피부 영역 내 픽셀만 고려
    p = prob * face_mask              # [B,1,H,W]
    g = gt   * face_mask              # [B,1,H,W]

    # 샘플별 합산 → [B]
    inter  = (p * g).flatten(1).sum(1)
    p_sum  = p.flatten(1).sum(1)
    g_sum  = g.flatten(1).sum(1)

    dice = 1.0 - (2.0 * inter + smooth) / (p_sum + g_sum + smooth)
    return dice                        # [B]


def _bce_per_pixel(pred:      torch.Tensor,
                   gt:        torch.Tensor,
                   face_mask: torch.Tensor) -> torch.Tensor:
    """BCE를 피부 영역에만 적용. [B,1,H,W] 반환.
    pred : logit (binary_cross_entropy_with_logits 사용 → 수치 안정)
    """
    bce = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    return bce * face_mask             # [B,1,H,W]


def _focal_per_pixel(pred:      torch.Tensor,
                     gt:        torch.Tensor,
                     face_mask: torch.Tensor,
                     gamma:     float = 2.0,
                     alpha:     float = 0.25) -> torch.Tensor:
    """
    Focal Loss를 피부 영역에만 적용. [B,1,H,W] 반환.

    pred  : logit (sigmoid 전) — binary_cross_entropy_with_logits 사용
    gamma : 어려운 샘플 집중 계수 (2.0 권장)
    alpha : 양성 가중치 (주름 희소성 보완)
    """
    bce   = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')  # [B,1,H,W]
    prob  = torch.sigmoid(pred)
    p_t   = torch.where(gt > 0.5, prob, 1.0 - prob)             # 예측 확률
    focal = alpha * (1.0 - p_t) ** gamma * bce
    return focal * face_mask           # [B,1,H,W]


# ══════════════════════════════════════════════════════════════════════════════
# SkinAnalyzerLoss
# ══════════════════════════════════════════════════════════════════════════════
class SkinAnalyzerLoss(nn.Module):

    _MEL_ABS = [0.28, 0.18, 0.09]
    _HEM_ABS = [0.10, 0.35, 0.05]

    def __init__(self,
                 w_brown:      float = 1.0,
                 w_red:        float = 1.0,
                 w_wrinkle:    float = 1.0,
                 w_recon:      float = 0.3,
                 w_consist:    float = 0.0,
                 w_freq_reg:   float = 0.0,
                 bce_dice_ratio: float = 0.5,   # BCE:Dice 비율 (brown/red)
                 focal_gamma:  float = 2.0,
                 focal_alpha:  float = 0.25):
        super().__init__()
        self.w_brown      = w_brown
        self.w_red        = w_red
        self.w_wrinkle    = w_wrinkle
        self.w_recon      = w_recon
        self.w_consist    = w_consist
        self.w_freq_reg   = w_freq_reg
        self.bce_ratio    = bce_dice_ratio   # α·BCE + (1-α)·Dice
        self.focal_gamma  = focal_gamma
        self.focal_alpha  = focal_alpha

        self.register_buffer('mel_abs',
                             torch.tensor(self._MEL_ABS).view(1, 3, 1, 1))
        self.register_buffer('hem_abs',
                             torch.tensor(self._HEM_ABS).view(1, 3, 1, 1))

    # ── brown / red : BCE + Dice ─────────────────────────────────────────────
    def _bce_dice(self,
                  pred:      torch.Tensor,
                  gt:        torch.Tensor,
                  face_mask: torch.Tensor,
                  has_gt:    list[bool]) -> torch.Tensor:
        """
        BCE + Dice (부분 GT 지원)

        pred / gt / face_mask : [B,1,H,W]
        has_gt                : list[bool] 길이 B
        """
        gt_avail = torch.tensor(has_gt, dtype=torch.float32,
                                device=pred.device).view(-1, 1, 1, 1)

        # ── BCE ───────────────────────────────────────────────────────────────
        bce_map = _bce_per_pixel(pred, gt, face_mask)              # [B,1,H,W]
        bce_map = bce_map * gt_avail
        denom_pix = (face_mask * gt_avail).sum().clamp(min=1.0)
        l_bce = bce_map.sum() / denom_pix

        # ── Dice ──────────────────────────────────────────────────────────────
        dice_per = _dice_loss_per_sample(pred, gt, face_mask)      # [B]
        avail_1d = torch.tensor(has_gt, dtype=torch.float32,
                                device=pred.device)
        n_valid  = avail_1d.sum().clamp(min=1.0)
        l_dice   = (dice_per * avail_1d).sum() / n_valid

        return self.bce_ratio * l_bce + (1.0 - self.bce_ratio) * l_dice

    # ── wrinkle : Focal + Dice ────────────────────────────────────────────────
    def _focal_dice(self,
                    pred:      torch.Tensor,
                    gt:        torch.Tensor,
                    face_mask: torch.Tensor,
                    has_gt:    list[bool]) -> torch.Tensor:
        """
        Focal + Dice (부분 GT 지원)

        pred / gt / face_mask : [B,1,H,W]
        has_gt                : list[bool] 길이 B
        """
        gt_avail = torch.tensor(has_gt, dtype=torch.float32,
                                device=pred.device).view(-1, 1, 1, 1)

        # ── Focal ─────────────────────────────────────────────────────────────
        focal_map = _focal_per_pixel(pred, gt, face_mask,
                                     self.focal_gamma, self.focal_alpha)
        focal_map = focal_map * gt_avail
        denom_pix = (face_mask * gt_avail).sum().clamp(min=1.0)
        l_focal   = focal_map.sum() / denom_pix

        # ── Dice ──────────────────────────────────────────────────────────────
        dice_per = _dice_loss_per_sample(pred, gt, face_mask)      # [B]
        avail_1d = torch.tensor(has_gt, dtype=torch.float32,
                                device=pred.device)
        n_valid  = avail_1d.sum().clamp(min=1.0)
        l_dice   = (dice_per * avail_1d).sum() / n_valid

        return self.bce_ratio * l_focal + (1.0 - self.bce_ratio) * l_dice

    # ── Consistency용 내부 L1 (augmentation 비교) ────────────────────────────
    def _partial_l1(self,
                    pred:      torch.Tensor,
                    gt:        torch.Tensor,
                    face_mask: torch.Tensor,
                    has_gt:    list[bool]) -> torch.Tensor:
        """조명 consistency loss에서 augmented 예측끼리 비교 시 사용.
        pred / gt : logit — sigmoid 공간에서 비교 (0~1 범위로 정규화)
        """
        gt_avail  = torch.tensor(has_gt, dtype=torch.float32,
                                 device=pred.device).view(-1, 1, 1, 1)
        per_pixel = F.l1_loss(torch.sigmoid(pred), torch.sigmoid(gt),
                              reduction='none') * face_mask * gt_avail
        denom     = (face_mask * gt_avail).sum().clamp(min=1.0)
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

        brown_prob = torch.sigmoid(brown_mask)
        red_prob   = torch.sigmoid(red_mask)
        od_recon  = brown_prob * self.mel_abs + red_prob * self.hem_abs
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

        # ── Supervised (부분) ────────────────────────────────────────────────
        # brown / red : BCE + Dice
        l_brown = self._bce_dice(
            result.brown_mask,   brown_gt,   face_mask, has_brown)
        l_red = self._bce_dice(
            result.red_mask,     red_gt,     face_mask, has_red)
        # wrinkle : Focal + Dice
        l_wrinkle = self._focal_dice(
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
