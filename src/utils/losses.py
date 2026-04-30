import torch
import torch.nn.functional as F


def bit_loss(bit_logits, bit_targets):
    """Binary cross-entropy loss for watermark bit prediction."""
    return F.binary_cross_entropy_with_logits(bit_logits, bit_targets.float())


def latent_loss(w_pred, w_true):
    """Mean-squared error between predicted and target watermark latents."""
    return F.mse_loss(w_pred, w_true)


def image_loss(img_pred, img_orig):
    """Image-space reconstruction loss used during watermark training."""
    return F.mse_loss(img_pred, img_orig)
