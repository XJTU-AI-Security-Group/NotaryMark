import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(4, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        if cond_dim is not None:
            self.cond_proj = nn.Linear(cond_dim, out_channels)
        else:
            self.cond_proj = None

    def forward(self, x, cond=None):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))

        if self.cond_proj is not None and cond is not None:
            # cond shape: [B, cond_dim]
            cond_emb = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
            h = h + cond_emb

        return h + self.shortcut(x)


class PredictorUNet(nn.Module):
    def __init__(self, latent_dim=4, base_channels=64, text_dim=768, depth=4):
        super().__init__()
        self.latent_dim = latent_dim

        # encoder
        self.encoders = nn.ModuleList()
        in_ch = latent_dim
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(ResidualBlock(in_ch, out_ch, cond_dim=text_dim))
            in_ch = out_ch

        # bottleneck
        self.bottleneck = ResidualBlock(in_ch, in_ch, cond_dim=text_dim)

        # decoder
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth-1)):
            out_ch = base_channels * (2 ** i)
            self.decoders.append(ResidualBlock(in_ch, out_ch, cond_dim=text_dim))
            in_ch = out_ch
        self.decoders.append(ResidualBlock(in_ch, out_ch, cond_dim=text_dim))

        self.out_conv = nn.Conv2d(in_ch, latent_dim, 3, padding=1)

    def forward(self, z_w, text_emb):
        """
        z_w: [B, latent_dim, H, W]
        text_emb: [B, text_dim]
        """
        skips = []
        h = z_w

        # encoder
        for enc in self.encoders:
            h = enc(h, cond=text_emb)
            skips.append(h)
            h = F.avg_pool2d(h, 2)


        h = self.bottleneck(h, cond=text_emb)

        # decoder
        for dec, skip in zip(self.decoders, reversed(skips)):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = h + skip
            h = dec(h, cond=text_emb)

        w_pred = self.out_conv(h)
        return w_pred
