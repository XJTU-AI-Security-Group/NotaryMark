import torch
import torch.nn as nn


class BitEncoder(nn.Module):
    """
    BitEncoder

    A lightweight MLP-based encoder that maps a binary payload (bit vector) into
    a latent tensor with the same shape convention as Stable Diffusion VAE latents.

    Input/Output:
        - Input bits:  [B, bit_length]  (typically 0/1 or in [0, 1])
        - Output latent w: [B, latent_dim, spatial_size, spatial_size]
          (default: [B, 4, 64, 64])

    Notes:
        - This module produces a *continuous* latent tensor. It does not apply any
          probability squashing (e.g., sigmoid) because it is not a classifier.
        - If you want to constrain the latent range, consider adding an activation
          (e.g., tanh) at the end, but that is task-dependent.

    Args:
        bit_length (int): Payload length (number of bits).
        latent_dim (int): Output latent channels (e.g., 4 for SD).
        spatial_size (int): Output spatial resolution (assumes square H=W).
    """

    def __init__(self, bit_length: int = 64, latent_dim: int = 4, spatial_size: int = 64):
        super().__init__()
        self.bit_length = bit_length
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size

        # First projection from bit space to a small feature embedding.
        self.fc = nn.Linear(bit_length, 512)

        # MLP head expands features to the full latent tensor size.
        self.net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim * spatial_size * spatial_size),
        )

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            bits (torch.Tensor):
                Payload tensor of shape [B, bit_length]. Can be 0/1 integers or floats.

        Returns:
            torch.Tensor:
                Encoded latent tensor w of shape [B, latent_dim, spatial_size, spatial_size].
        """
        # Ensure floating point for linear layers: [B, bit_length] -> [B, 512]
        x = self.fc(bits.float())

        # Expand to flattened latent: [B, 512] -> [B, latent_dim*H*W]
        x = self.net(x)

        # Reshape to latent map: [B, latent_dim, H, W]
        return x.view(-1, self.latent_dim, self.spatial_size, self.spatial_size)


class WatermarkEncoder(nn.Module):
    """
    WatermarkEncoder

    A deconvolutional (transpose-conv) encoder that maps a bit vector into a
    spatial latent feature map (e.g., SD-style latent) through progressive upsampling.

    Architecture:
        1) A linear layer expands the bit vector to a small feature map: [B, 512, 4, 4]
        2) Four ConvTranspose2d blocks upsample by factor 2 each time:
            4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        3) Final output has `channels` channels, typically 4 for SD latents.

    Input/Output:
        - Input bits/message w:  [B, bit_length]
        - Output latent z:       [B, channels, 64, 64]  (by default)

    Why no Sigmoid?
        - This module is a *generator/encoder* producing continuous latents, not
          a probability output. Sigmoid would unnecessarily restrict the output
          to [0, 1] and may hurt representational capacity.
        - If a downstream decoder predicts bits, *that* decoder should output logits
          and be trained with `nn.BCEWithLogitsLoss` (no sigmoid inside the decoder).

    Why Tanh at the end?
        - Tanh normalizes the latent output to [-1, 1], which can stabilize training
          and prevent exploding activations when the generated latent is used as a
          watermark carrier or combined with other latent signals.
        - Whether [-1, 1] is ideal depends on your pipeline. If you need to match a
          specific latent distribution (e.g., SD VAE scaling), consider calibrating
          with an additional scale factor or normalization strategy.

    Training recommendations (typical):
        - If paired with a bit-decoder that outputs logits:
            loss_bits = nn.BCEWithLogitsLoss()(logits, target_bits.float())
        - For inference on decoder side:
            probs = torch.sigmoid(logits); bits_hat = (probs > 0.5).int()

    Args:
        bit_length (int): Payload length (number of bits).
        channels (int): Output latent channels (e.g., 4 for SD).
        height (int): Target output height (kept for clarity; default 64).
        width (int): Target output width  (kept for clarity; default 64).
    """

    def __init__(self, bit_length: int, channels: int = 4, height: int = 64, width: int = 64):
        super().__init__()
        self.bit_length = bit_length
        self.channels = channels
        self.height = height
        self.width = width

        # Expand the payload into a compact "seed" feature map.
        # Target seed shape: [B, 512, 4, 4]
        self.initial_dense = nn.Linear(bit_length, 512 * 4 * 4)

        # Progressive upsampling with transpose convolutions.
        # Each block doubles spatial resolution (stride=2).
        self.up_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, self.channels, kernel_size=4, stride=2, padding=1),  # [B, C, 64, 64]

            # Normalize output range for stability. This is *not* a probability.
            nn.Tanh(),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            w (torch.Tensor):
                Payload tensor of shape [B, bit_length].

        Returns:
            torch.Tensor:
                Generated latent tensor z of shape [B, channels, 64, 64] (default).
        """
        # Linear expansion: [B, bit_length] -> [B, 512*4*4]
        x = self.initial_dense(w.float())

        # Reshape to seed feature map: [B, 512, 4, 4]
        x = x.view(-1, 512, 4, 4)

        # Upsample to the target latent resolution: [B, channels, 64, 64]
        z = self.up_blocks(x)
        return z
