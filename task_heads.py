"""
task_heads.py
=============
Task-specific prediction heads

SpotHead   : 갈색 반점 / 적색 반점 예측
             입력: encoder의 chroma skip features + bottleneck
             출력: mask [B,1,H,W], score [B]  (0~100)

WrinkleHead: 주름 예측
             입력: encoder의 texture skip features
             출력: mask [B,1,H,W], score [B]  (0~100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 공통 빌딩블록
# ══════════════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    """Conv3×3 → BN → ReLU × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """ConvTranspose2d(×2) → concat skip → ConvBlock"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# 스코어 계산 유틸
# ══════════════════════════════════════════════════════════════════════════════
def _compute_score(mask: torch.Tensor,
                   skin_mask: torch.Tensor | None) -> torch.Tensor:
    """
    mask      : [B, 1, H, W]  sigmoid 출력 (0~1)
    skin_mask : [B, 1, H, W]  피부 영역 마스크 (없으면 전체)
    returns   : [B]  스코어 0~100
    """
    if skin_mask is None:
        skin_mask = torch.ones_like(mask)

    denom      = skin_mask.sum(dim=[1, 2, 3]).clamp(min=1.0)
    area_ratio = (mask * skin_mask).sum(dim=[1, 2, 3]) / denom   # 면적 비율
    density    = (mask * skin_mask).flatten(1).max(dim=1).values  # 최대 강도

    score = (area_ratio * 0.6 + density * 0.4) * 100.0
    return score.clamp(0.0, 100.0)                                 # [B]


# ══════════════════════════════════════════════════════════════════════════════
# SpotHead  (갈색 반점 / 적색 반점)
# ══════════════════════════════════════════════════════════════════════════════
class SpotHead(nn.Module):
    """
    발색단 반점 예측 (brown spot 또는 red spot)

    입력
    ----
    chroma_feats : [c1, c2, c3]  — encoder mid-freq skips
                    c1: [B, ch,   H,   W]
                    c2: [B, ch*2, H/2, H/2]
                    c3: [B, ch*4, H/4, H/4]
    bottleneck   : [B, ch*8, H/8, H/8]
    skin_mask    : [B, 1, H, W]  (optional)

    출력
    ----
    mask  : [B, 1, H, W]  sigmoid → 반점 확률 맵
    score : [B]           0~100 스코어
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        ch = base_ch

        # Decoder: bottleneck → d3 → d2 → d1
        self.up3 = UpBlock(ch * 8, ch * 4, ch * 4)
        self.up2 = UpBlock(ch * 4, ch * 2, ch * 2)
        self.up1 = UpBlock(ch * 2, ch,     ch)

        # 예측 head
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 1, kernel_size=1),
        )

    def forward(self,
                chroma_feats: list,
                bottleneck: torch.Tensor,
                skin_mask: torch.Tensor | None = None):
        c1, c2, c3 = chroma_feats   # fine → coarse

        d3 = self.up3(bottleneck, c3)
        d2 = self.up2(d3,         c2)
        d1 = self.up1(d2,         c1)

        mask  = torch.sigmoid(self.head(d1))   # [B, 1, H, W]
        score = _compute_score(mask, skin_mask) # [B]
        return mask, score


# ══════════════════════════════════════════════════════════════════════════════
# WrinkleHead
# ══════════════════════════════════════════════════════════════════════════════
class WrinkleHead(nn.Module):
    """
    주름 예측

    고주파 텍스처 feature를 multi-scale로 concat 후 예측.
    주름은 고주파 성분이므로 texture skips를 사용.

    입력
    ----
    texture_feats : [t1, t2, t3]  — encoder high-freq skips
                    t1: [B, ch,   H,   W]
                    t2: [B, ch*2, H/2, H/2]
                    t3: [B, ch*4, H/4, H/4]
    skin_mask     : [B, 1, H, W]  (optional)

    출력
    ----
    mask  : [B, 1, H, W]
    score : [B]
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        ch = base_ch
        fused_ch = ch + ch * 2 + ch * 4   # 모든 레벨 concat

        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 1, kernel_size=1),
        )

    def forward(self,
                texture_feats: list,
                skin_mask: torch.Tensor | None = None):
        t1, t2, t3 = texture_feats   # fine → coarse

        # 모든 레벨을 t1 해상도로 업샘플 후 concat
        target_h, target_w = t1.shape[2], t1.shape[3]
        t2_up = F.interpolate(t2, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)
        t3_up = F.interpolate(t3, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)

        fused = torch.cat([t1, t2_up, t3_up], dim=1)   # [B, fused_ch, H, W]
        mask  = torch.sigmoid(self.fuse(fused))          # [B, 1, H, W]
        score = _compute_score(mask, skin_mask)           # [B]
        return mask, score
