import torch
from diffusers import AutoencoderKL


class VAEWrapper:
    """Thin wrapper around Stable Diffusion's VAE for latent-space training."""

    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
        self.vae.eval()

    @torch.no_grad()
    def encode(self, images):
        """
        Encode images into Stable Diffusion latents.

        Args:
            images: Tensor [B, 3, H, W] in pixel space with values in [0, 1].

        Returns:
            Latent tensor [B, latent_dim, H/8, W/8].
        """
        images = images * 2 - 1  # [0,1] -> [-1,1]
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # Stable Diffusion latent scaling factor.
        return latents

    @torch.no_grad()
    def decode(self, latents):
        """
        Decode latents back into image space.

        Args:
            latents: Tensor [B, latent_dim, H/8, W/8].

        Returns:
            Reconstructed images [B, 3, H, W].
        """
        latents = latents / 0.18215
        images = self.vae.decode(latents).sample
        images = (images.clamp(-1, 1) + 1) / 2
        return images
