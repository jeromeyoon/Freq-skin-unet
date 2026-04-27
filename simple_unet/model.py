import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DecodeBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EfficientNetUNet(nn.Module):
    """
    EfficientNet-B4 encoder + UNet decoder → binary mask [B,1,H,W]

    EfficientNet-B4 features_only channels:
      feat[0]: 24ch, stride /2
      feat[1]: 32ch, stride /4
      feat[2]: 56ch, stride /8
      feat[3]: 160ch, stride /16
      feat[4]: 448ch, stride /32
    """

    ENC_CHANNELS = [24, 32, 56, 160, 448]
    DEC_CHANNELS = [256, 128, 64, 32]

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            'efficientnet_b4',
            features_only=True,
            pretrained=pretrained,
        )

        enc = self.ENC_CHANNELS
        dec = self.DEC_CHANNELS

        self.up4 = DecodeBlock(enc[4], enc[3], dec[0])
        self.up3 = DecodeBlock(dec[0], enc[2], dec[1])
        self.up2 = DecodeBlock(dec[1], enc[1], dec[2])
        self.up1 = DecodeBlock(dec[2], enc[0], dec[3])

        self.head = nn.Conv2d(dec[3], 1, kernel_size=1)

    def forward(self, x):
        # x: [B, 3, H, W]
        f0, f1, f2, f3, f4 = self.encoder(x)

        d = self.up4(f4, f3)
        d = self.up3(d, f2)
        d = self.up2(d, f1)
        d = self.up1(d, f0)

        d = F.interpolate(d, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(self.head(d))   # [B, 1, H, W]
