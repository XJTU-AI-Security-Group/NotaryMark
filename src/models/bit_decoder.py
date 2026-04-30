import torch
import torch.nn as nn


class BitDecoder(nn.Module):
    """
    BitDecoder

    A simple MLP-based decoder that flattens a latent tensor and predicts a
    multi-bit watermark as logits.

    This module is intended to map a latent feature map (e.g., VAE latent)
    of shape [B, C, H, W] to a bit vector of shape [B, bit_length].

    Notes:
        - The output is *logits* (unnormalized scores). For training, it is
          recommended to use `nn.BCEWithLogitsLoss`, which is numerically stable.
        - For inference, apply `torch.sigmoid(logits)` to get probabilities,
          then threshold at 0.5 to obtain binary bits.

    Args:
        bit_length (int): Number of bits to decode (payload length).
        latent_dim (int): Channel dimension C of the latent tensor.
        spatial_size (int): Spatial resolution (assumes H=W=spatial_size).
    """

    def __init__(self, bit_length: int = 64, latent_dim: int = 4, spatial_size: int = 64):
        super().__init__()
        in_dim = latent_dim * spatial_size * spatial_size

        # First projection after flattening to obtain a compact representation.
        self.fc = nn.Linear(in_dim, 1024)

        # Lightweight MLP head that maps features to bit logits.
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, bit_length),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            w (torch.Tensor):
                Latent tensor of shape [B, latent_dim, spatial_size, spatial_size].

        Returns:
            torch.Tensor:
                Bit logits of shape [B, bit_length]. These are raw logits
                (NOT passed through sigmoid).
        """
        # Flatten spatial dimensions: [B, C, H, W] -> [B, C*H*W]
        x = w.view(w.size(0), -1)

        # Project to hidden feature space
        x = self.fc(x)

        # Predict bit logits
        return self.net(x)


class WatermarkDecoder(nn.Module):
    """
    WatermarkDecoder

    A convolutional decoder that extracts hierarchical features from a latent
    tensor via strided downsampling and predicts a multi-bit watermark as logits.

    Architecture:
        - 4 strided Conv2d blocks (stride=2) progressively downsample:
            64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        - A small MLP head maps the final 4x4 feature map to `bit_length` logits.

    Why logits (no Sigmoid)?
        - This decoder returns logits so you can train with `nn.BCEWithLogitsLoss`,
          which combines a sigmoid layer and BCE loss in a numerically stable way.
        - During inference, use `torch.sigmoid(logits)` to obtain probabilities.

    Args:
        bit_length (int): Number of bits to decode (payload length).
        channels (int): Input latent channels (e.g., 4 for Stable Diffusion VAE latents).
        height (int): Input height (kept for clarity; network expects 64 by default).
        width (int): Input width  (kept for clarity; network expects 64 by default).
    """

    def __init__(self, bit_length: int, channels: int = 4, height: int = 64, width: int = 64):
        super().__init__()

        # Strided convolutions for feature extraction + downsampling.
        # Input:  [B, channels, 64, 64]
        # Output: [B, 256,      4,  4]
        self.down_blocks = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final MLP head to map flattened features to watermark bit logits.
        self.final_dense = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, bit_length),
            # NOTE: No sigmoid here. Use BCEWithLogitsLoss during training.
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z (torch.Tensor):
                Latent tensor of shape [B, channels, height, width].
                Typically [B, 4, 64, 64] for VAE latents.

        Returns:
            torch.Tensor:
                Watermark bit logits of shape [B, bit_length].
                Apply sigmoid for probabilities if needed.
        """
        # Convolutional feature extraction + downsampling
        x = self.down_blocks(z)

        # Flatten: [B, 256, 4, 4] -> [B, 256*4*4]
        x = x.view(x.size(0), -1)

        # Predict bit logits
        w_prime_logits = self.final_dense(x)
        return w_prime_logits
