"""
skin_loss_smooth.py
===================
SkinAnalyzerLoss + 개선된 get_loss_weights (skin_loss.py 대체)

변경사항 (vs skin_loss.py):
  1. get_loss_weights: Phase 전환 시 가중치를 즉시 점프하지 않고 warm-up 구간에서
     선형 보간으로 점진 증가 (학습 불안정 방지)
  2. w_freq_reg를 Phase 0부터 소량(0.02) 적용하여 FrequencyGate가 초기에
     무너지는 것을 방지

Phase 스케줄 (전체 에폭 기준):
  Phase 0 (0~20%):     supervised only
                        w_freq_reg=0.02 (gate 조기 보호)
  Phase 1 (20~40%):    + recon
                        w_freq_reg=0.02
  Phase 2 warmup (40~47%): + consistency 선형 증가 0→0.3
  Phase 2 stable (47~80%): consistency=0.3
  Phase 3 warmup (80~87%): + freq_reg 선형 증가 0.02→0.1
  Phase 3 stable (87~100%): freq_reg=0.1

SkinAnalyzerLoss 클래스는 skin_loss.py와 동일 (변경 없음).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from skin_net import SkinResult


# ══════════════════════════════════════════════════════════════════════════════
# 기본 Loss 함수 (변경 없음)
# ══════════════════════════════════════════════════════════════════════════════
def _dice_loss_per_sample(pred, gt, face_mask, smooth=1.0):
    prob = torch.sigmoid(pred)
    p = prob * face_mask
    g = gt   * face_mask
    inter  = (p * g).flatten(1).sum(1)
    p_sum  = p.flatten(1).sum(1)
    g_sum  = g.flatten(1).sum(1)
    return 1.0 - (2.0 * inter + smooth) / (p_sum + g_sum + smooth)


def _bce_per_pixel(pred, gt, face_mask, pos_weight=None):
    bce = F.binary_cross_entropy_with_logits(
        pred, gt, pos_weight=pos_weight, reduction='none')
    return bce * face_mask


def _focal_per_pixel(pred, gt, face_mask, gamma=2.0, alpha=0.25):
    bce   = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    prob  = torch.sigmoid(pred)
    p_t   = torch.where(gt > 0.5, prob, 1.0 - prob)
    focal = alpha * (1.0 - p_t) ** gamma * bce
    return focal * face_mask


# ══════════════════════════════════════════════════════════════════════════════
# SkinAnalyzerLoss (변경 없음)
# ══════════════════════════════════════════════════════════════════════════════
class SkinAnalyzerLoss(nn.Module):

    _MEL_ABS = [0.28, 0.18, 0.09]
    _HEM_ABS = [0.10, 0.35, 0.05]

    def __init__(self,
                 w_brown=1.0, w_red=1.0, w_wrinkle=1.0,
                 w_recon=0.3, w_consist=0.0, w_freq_reg=0.0,
                 bce_dice_ratio=0.3, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.w_brown      = w_brown
        self.w_red        = w_red
        self.w_wrinkle    = w_wrinkle
        self.w_recon      = w_recon
        self.w_consist    = w_consist
        self.w_freq_reg   = w_freq_reg
        self.bce_ratio    = bce_dice_ratio
        self.focal_gamma  = focal_gamma
        self.focal_alpha  = focal_alpha

        self.register_buffer('mel_abs',
                             torch.tensor(self._MEL_ABS).view(1, 3, 1, 1))
        self.register_buffer('hem_abs',
                             torch.tensor(self._HEM_ABS).view(1, 3, 1, 1))

    def _focal_dice_spot(self, pred, gt, face_mask, has_gt, alpha=0.5):
        gt_avail = torch.tensor(has_gt, dtype=torch.float32,
                                device=pred.device).view(-1, 1, 1, 1)
        focal_map = _focal_per_pixel(pred, gt, face_mask, self.focal_gamma, alpha)
        focal_map = focal_map * gt_avail
        denom_pix = (face_mask * gt_avail).sum().clamp(min=1.0)
        l_focal   = focal_map.sum() / denom_pix

        dice_per = _dice_loss_per_sample(pred, gt, face_mask)
        avail_1d = torch.tensor(has_gt, dtype=torch.float32, device=pred.device)
        n_valid  = avail_1d.sum().clamp(min=1.0)
        l_dice   = (dice_per * avail_1d).sum() / n_valid

        return self.bce_ratio * l_focal + (1.0 - self.bce_ratio) * l_dice

    def _focal_dice(self, pred, gt, face_mask, has_gt):
        gt_avail = torch.tensor(has_gt, dtype=torch.float32,
                                device=pred.device).view(-1, 1, 1, 1)
        focal_map = _focal_per_pixel(pred, gt, face_mask,
                                     self.focal_gamma, self.focal_alpha)
        focal_map = focal_map * gt_avail
        denom_pix = (face_mask * gt_avail).sum().clamp(min=1.0)
        l_focal   = focal_map.sum() / denom_pix

        dice_per = _dice_loss_per_sample(pred, gt, face_mask)
        avail_1d = torch.tensor(has_gt, dtype=torch.float32, device=pred.device)
        n_valid  = avail_1d.sum().clamp(min=1.0)
        l_dice   = (dice_per * avail_1d).sum() / n_valid

        return self.bce_ratio * l_focal + (1.0 - self.bce_ratio) * l_dice

    def _partial_l1(self, pred, gt, face_mask, has_gt):
        gt_avail  = torch.tensor(has_gt, dtype=torch.float32,
                                 device=pred.device).view(-1, 1, 1, 1)
        per_pixel = F.l1_loss(torch.sigmoid(pred), torch.sigmoid(gt),
                              reduction='none') * face_mask * gt_avail
        denom = (face_mask * gt_avail).sum().clamp(min=1.0)
        return per_pixel.sum() / denom

    def _beer_lambert_recon(self, brown_mask, red_mask, rgb_cross,
                            face_mask, has_brown, has_red):
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

    @staticmethod
    def _freq_reg(model):
        enc = model.cross_encoder
        gates = [
            torch.sigmoid(enc.freq1.low_logit).mean(),
            torch.sigmoid(enc.freq2.low_logit).mean(),
            torch.sigmoid(enc.freq3.low_logit).mean(),
        ]
        return sum(gates) / len(gates)

    def forward(self, result, brown_gt, red_gt, wrinkle_gt,
                rgb_cross, face_mask,
                has_brown, has_red, has_wrinkle,
                result_aug=None, model=None):
        detail = {}

        l_brown   = self._focal_dice_spot(result.brown_mask,   brown_gt,   face_mask, has_brown, alpha=0.5)
        l_red     = self._focal_dice_spot(result.red_mask,     red_gt,     face_mask, has_red,   alpha=0.5)
        l_wrinkle = self._focal_dice(     result.wrinkle_mask, wrinkle_gt, face_mask, has_wrinkle)
        detail.update(brown=l_brown.item(), red=l_red.item(), wrinkle=l_wrinkle.item())

        loss = torch.tensor(0.0, device=rgb_cross.device)
        if any(has_brown):   loss = loss + self.w_brown   * l_brown
        if any(has_red):     loss = loss + self.w_red     * l_red
        if any(has_wrinkle): loss = loss + self.w_wrinkle * l_wrinkle

        if self.w_recon > 0:
            l_recon = self._beer_lambert_recon(
                result.brown_mask, result.red_mask, rgb_cross, face_mask, has_brown, has_red)
            detail['recon'] = l_recon.item()
            loss = loss + self.w_recon * l_recon
        else:
            detail['recon'] = 0.0

        if self.w_consist > 0 and result_aug is not None:
            l_consist = (
                self._partial_l1(result.brown_mask,   result_aug.brown_mask.detach(),   face_mask, has_brown) +
                self._partial_l1(result.red_mask,     result_aug.red_mask.detach(),     face_mask, has_red)   +
                self._partial_l1(result.wrinkle_mask, result_aug.wrinkle_mask.detach(), face_mask, has_wrinkle)
            ) / 3.0
            detail['consist'] = l_consist.item()
            loss = loss + self.w_consist * l_consist
        else:
            detail['consist'] = 0.0

        if self.w_freq_reg > 0 and model is not None:
            l_freq = self._freq_reg(model)
            detail['freq_reg'] = l_freq.item()
            loss = loss + self.w_freq_reg * l_freq
        else:
            detail['freq_reg'] = 0.0

        return loss, detail


# ══════════════════════════════════════════════════════════════════════════════
# get_loss_weights — smooth 버전 (핵심 변경)
# ══════════════════════════════════════════════════════════════════════════════
def _lerp(a: float, b: float, t: float) -> float:
    """선형 보간: t=0→a, t=1→b"""
    return a + (b - a) * t


def get_loss_weights(epoch:        int,
                     total_epochs: int,
                     w_brown:      float = 1.0,
                     w_red:        float = 1.0,
                     w_wrinkle:    float = 1.0,
                     w_recon:      float = 0.3) -> dict:
    """
    단계별 loss 가중치 반환 (curriculum + smooth auxiliary).

    난이도 가정:
      - brown   : 가장 쉬움
      - red     : 중간
      - wrinkle : 가장 어려움

    Curriculum 철학:
      - 초반(0~20%)  : brown 중심으로 안정 수렴
      - 중반(20~50%): brown → red/wrinkle로 점진 이동
      - 후반(50~80%): wrinkle 강화 시작
      - 말기(80~100%): wrinkle 집중 미세 조정

    Auxiliary 스케줄은 smooth 버전 철학 유지:
      [0%, 20%)    supervised only, w_recon=0, w_freq_reg=0.02
      [20%, 40%)   + recon(0.3),   w_freq_reg=0.02
      [40%, 47%)   consistency 0→0.3 선형 증가
      [47%, 80%)   consistency=0.3
      [80%, 87%)   freq_reg 0.02→0.1 선형 증가
      [87%, 100%)  freq_reg=0.1
    """
    progress = epoch / max(total_epochs, 1)

    # ── task curriculum ────────────────────────────────────────────────────
    if progress < 0.20:
        task = dict(
            w_brown   = w_brown * 1.4,
            w_red     = w_red * 0.8,
            w_wrinkle = w_wrinkle * 0.4,
        )
    elif progress < 0.50:
        t = (progress - 0.20) / 0.30
        task = dict(
            w_brown   = _lerp(w_brown * 1.4,      w_brown * 1.0, t),
            w_red     = _lerp(w_red * 0.8,        w_red * 1.0,   t),
            w_wrinkle = _lerp(w_wrinkle * 0.4,    w_wrinkle * 0.9, t),
        )
    elif progress < 0.80:
        t = (progress - 0.50) / 0.30
        task = dict(
            w_brown   = _lerp(w_brown * 1.0,      w_brown * 0.85, t),
            w_red     = _lerp(w_red * 1.0,        w_red * 1.05,   t),
            w_wrinkle = _lerp(w_wrinkle * 0.9,    w_wrinkle * 1.3, t),
        )
    else:
        t = (progress - 0.80) / 0.20
        task = dict(
            w_brown   = _lerp(w_brown * 0.85,     w_brown * 0.75, t),
            w_red     = _lerp(w_red * 1.05,       w_red * 1.0,    t),
            w_wrinkle = _lerp(w_wrinkle * 1.3,    w_wrinkle * 1.5, t),
        )

    # ── auxiliary curriculum ───────────────────────────────────────────────
    base_aux = dict(
        w_recon    = w_recon,
        w_consist  = 0.0,
        w_freq_reg = 0.02,
    )

    # Phase 0: supervised only (recon off)
    if progress < 0.20:
        aux = {**base_aux, 'w_recon': 0.0}

    # Phase 1: supervised + recon
    elif progress < 0.40:
        aux = base_aux

    # Phase 2 warm-up (40~47%): consistency 선형 증가
    elif progress < 0.47:
        t = (progress - 0.40) / 0.07          # 0→1
        aux = {**base_aux, 'w_consist': _lerp(0.0, 0.3, t)}

    # Phase 2 stable (47~80%)
    elif progress < 0.80:
        aux = {**base_aux, 'w_consist': 0.3}

    # Phase 3 warm-up (80~87%): freq_reg 선형 증가
    elif progress < 0.87:
        t = (progress - 0.80) / 0.07          # 0→1
        aux = {**base_aux, 'w_consist': 0.3, 'w_freq_reg': _lerp(0.02, 0.1, t)}

    # Phase 3 stable (87~100%)
    else:
        aux = {**base_aux, 'w_consist': 0.3, 'w_freq_reg': 0.1}

    return {**task, **aux}
