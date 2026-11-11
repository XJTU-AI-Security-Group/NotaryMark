#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pandas as pd


# Standard SSIM constants for data_range = 1.0 (Wang et al., 2004).
SSIM_C1 = 0.01 ** 2
SSIM_C2 = 0.03 ** 2

# Matches the defaults used by the reference implementation.
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def get_args():
    """
    Parse command-line arguments.

    This script measures the *fidelity* cost of watermark embedding, that is how much
    the watermarked images deviate from the originals. It is the natural companion of
    `embed_watermarks.py`, whose `--a` strength trades decoding robustness against
    visual distortion.

    Args:
        --original_dir: Folder of original (clean) images.
        --watermarked_dir: Folder of watermarked images produced by embed_watermarks.py.
        --output_csv: Path to save per-image metrics.
        --device: Torch device used for the metric computation.
        --self_test: Run analytic self-checks and exit (no data required).

    Returns:
        argparse.Namespace: Parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR / SSIM between original and watermarked image folders."
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        default="",
        help="Path to the folder of original images.",
    )
    parser.add_argument(
        "--watermarked_dir",
        type=str,
        default="",
        help="Path to the folder of watermarked images.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="fidelity_metrics.csv",
        help="Path to save the per-image metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for the metric computation.",
    )
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Run analytic self-checks of the PSNR / SSIM implementations and exit.",
    )
    return parser.parse_args()


def build_gaussian_kernel(
    window_size: int, sigma: float, channels: int, device: str, dtype: torch.dtype
) -> torch.Tensor:
    """
    Build a separable Gaussian window as a depthwise convolution kernel.

    The window follows the classic SSIM formulation (11 x 11, sigma = 1.5) and is
    normalized to sum to 1.

    Args:
        window_size (int): Odd spatial size of the window.
        sigma (float): Standard deviation of the Gaussian.
        channels (int): Number of channels; the kernel is applied per channel.
        device (str): Torch device.
        dtype (torch.dtype): Torch dtype (float64 is used for stable accumulation).

    Returns:
        torch.Tensor: Depthwise kernel of shape [channels, 1, window_size, window_size].
    """
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    gaussian_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    # Outer product gives a normalized 2-D window.
    window_2d = torch.outer(gaussian_1d, gaussian_1d)
    return window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def psnr(image: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Compute the peak signal-to-noise ratio between two images in [0, 1].

    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W] in [0, 1].
        reference (torch.Tensor): Reference tensor of the same shape and range.

    Returns:
        float: PSNR in dB, or `inf` when the two images are identical.
    """
    mse = torch.mean((image - reference) ** 2)
    if mse == 0:
        return float("inf")
    return float(-10.0 * torch.log10(mse))


def ssim(
    image: torch.Tensor,
    reference: torch.Tensor,
    window_size: int = SSIM_WINDOW_SIZE,
    sigma: float = SSIM_SIGMA,
) -> float:
    """
    Compute the structural similarity index between two images in [0, 1].

    SSIM is evaluated per channel with a Gaussian window and then averaged over
    channels. Borders are cropped by `window_size // 2` so the reported value covers
    only positions where the window is fully inside the image, which is what the
    reference implementation reports as well.

    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W] in [0, 1].
        reference (torch.Tensor): Reference tensor of the same shape and range.
        window_size (int): Odd spatial size of the Gaussian window.
        sigma (float): Standard deviation of the Gaussian window.

    Returns:
        float: Mean SSIM over the RGB channels.

    Raises:
        ValueError: If the images are smaller than the SSIM window.
    """
    _, height, width = image.shape
    if height < window_size or width < window_size:
        raise ValueError(
            f"Images must be at least {window_size}x{window_size} for SSIM; "
            f"got {height}x{width}."
        )

    channels = image.shape[0]
    kernel = build_gaussian_kernel(
        window_size, sigma, channels, device=str(image.device), dtype=image.dtype
    )

    # All five local statistics are filtered with the same kernel, so stack them into
    # one depthwise convolution instead of running five separate ones.
    #   [mu_x, mu_y, mu_xx, mu_yy, mu_xy]
    stacked = torch.cat(
        [
            image,
            reference,
            image * image,
            reference * reference,
            image * reference,
        ],
        dim=0,
    ).unsqueeze(0)  # [1, 5C, H, W]

    pad = window_size // 2
    # `F.conv2d` has no padding_mode argument, so pad by replication first and then
    # convolve with zero padding; the borders are cropped off right afterwards.
    padded = F.pad(stacked, (pad, pad, pad, pad), mode="replicate")
    filtered = F.conv2d(
        padded,
        kernel.repeat(5, 1, 1, 1),
        groups=5 * channels,
    )[0]

    # Drop the padded border so every retained position uses real pixels only.
    filtered = filtered[:, pad: height - pad, pad: width - pad]

    mu_x, mu_y, mu_xx, mu_yy, mu_xy = filtered.split(channels, dim=0)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy_prod = mu_x * mu_y

    sigma_x_sq = mu_xx - mu_x_sq
    sigma_y_sq = mu_yy - mu_y_sq
    sigma_xy = mu_xy - mu_xy_prod

    numerator = (2.0 * mu_xy_prod + SSIM_C1) * (2.0 * sigma_xy + SSIM_C2)
    denominator = (mu_x_sq + mu_y_sq + SSIM_C1) * (sigma_x_sq + sigma_y_sq + SSIM_C2)

    return float((numerator / denominator).mean())


def load_image(image_path: str, device: str) -> torch.Tensor:
    """
    Load an image as a float64 tensor in [0, 1] with shape [C, H, W].

    Args:
        image_path (str): Path to the image file.
        device (str): Torch device.

    Returns:
        torch.Tensor: Image tensor of shape [3, H, W] in [0, 1].
    """
    image = Image.open(image_path).convert("RGB")
    # uint8 -> float64 in one shot avoids an intermediate float32 copy.
    array = np.asarray(image, dtype="uint8").astype("float64") / 255.0
    # PIL yields [H, W, C]; the metrics expect [C, H, W].
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def index_images(folder: str) -> Dict[str, str]:
    """
    Map each image stem in a folder to its path.

    Args:
        folder (str): Folder to scan.

    Returns:
        Dict[str, str]: Mapping from file stem to image path.
    """
    indexed: Dict[str, str] = {}
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        stem = os.path.splitext(filename)[0]
        if stem in indexed:
            print(f"[WARN] Duplicate stem '{stem}' in {folder}; keeping {indexed[stem]}.")
            continue
        indexed[stem] = os.path.join(folder, filename)
    return indexed


def run_self_test() -> None:
    """
    Check the metric implementations against closed-form values.

    The cases use synthetic tensors, so no dataset and no checkpoint are needed.

    Raises:
        AssertionError: If any metric deviates from its analytic value.
    """
    device = "cpu"
    shape = (3, 32, 32)

    identical = torch.full(shape, 0.25, dtype=torch.float64, device=device)
    assert psnr(identical, identical) == float("inf"), "PSNR of identical images must be inf"
    assert abs(ssim(identical, identical) - 1.0) < 1e-12, "SSIM of identical images must be 1"

    # A constant offset of 0.5 everywhere gives MSE = 0.25, hence PSNR = 6.0206 dB.
    flat_a = torch.full(shape, 0.0, dtype=torch.float64, device=device)
    flat_b = torch.full(shape, 0.5, dtype=torch.float64, device=device)
    assert abs(psnr(flat_b, flat_a) - 6.020599913279624) < 1e-9, "PSNR closed form mismatch"

    # For constant images a and b the SSIM map collapses to (2ab + C1) / (a^2 + b^2 + C1).
    low = torch.full(shape, 0.4, dtype=torch.float64, device=device)
    high = torch.full(shape, 0.6, dtype=torch.float64, device=device)
    expected = (2 * 0.4 * 0.6 + SSIM_C1) / (0.4 ** 2 + 0.6 ** 2 + SSIM_C1)
    assert abs(ssim(high, low) - expected) < 1e-12, "SSIM closed form mismatch"

    print("[OK] self-test passed: PSNR / SSIM match their closed-form values.")


def main(args) -> None:
    """
    Compute PSNR and SSIM for every original/watermarked image pair and save the results.

    Images are paired by file stem, so the watermarked folder produced by
    `embed_watermarks.py` can be compared directly with its input folder. Pairs whose
    spatial size differs are reported and skipped instead of being silently resized,
    because resizing would corrupt the measured fidelity.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = args.device

    # -------------------------
    # Validate configuration
    # -------------------------
    if not os.path.isdir(args.original_dir):
        raise FileNotFoundError(f"Original directory not found: {args.original_dir}")
    if not os.path.isdir(args.watermarked_dir):
        raise FileNotFoundError(f"Watermarked directory not found: {args.watermarked_dir}")

    originals = index_images(args.original_dir)
    watermarked = index_images(args.watermarked_dir)

    stems = [stem for stem in originals if stem in watermarked]
    missing = sorted(set(originals) - set(watermarked))
    unexpected = sorted(set(watermarked) - set(originals))

    print(f"Original images: {len(originals)}")
    print(f"Watermarked images: {len(watermarked)}")
    if missing:
        print(f"[WARN] {len(missing)} original image(s) have no watermarked counterpart:", missing[:10])
    if unexpected:
        print(f"[WARN] {len(unexpected)} watermarked image(s) have no original counterpart:", unexpected[:10])

    if not stems:
        raise RuntimeError(
            "No image pairs could be matched by file stem; check --original_dir and --watermarked_dir."
        )

    # -------------------------
    # Score every pair
    # -------------------------
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for stem in tqdm(stems, desc="Scoring image pairs"):
        original = load_image(originals[stem], device=device)
        candidate = load_image(watermarked[stem], device=device)

        if original.shape != candidate.shape:
            print(
                f"\n[WARN] Skipping {stem}: size {tuple(original.shape[1:])} vs "
                f"{tuple(candidate.shape[1:])}. Re-run embedding at the original resolution."
            )
            skipped.append(stem)
            continue

        rows.append(
            {
                "filename": stem,
                "height": int(original.shape[1]),
                "width": int(original.shape[2]),
                "psnr_db": psnr(candidate, original),
                "ssim": ssim(candidate, original),
            }
        )

    if not rows:
        raise RuntimeError("Every matched pair was skipped; nothing to report.")

    # -------------------------
    # Summarize and save outputs
    # -------------------------
    frame = pd.DataFrame(rows)
    frame = frame[["filename", "height", "width", "psnr_db", "ssim"]]

    psnr_series = frame["psnr_db"].replace([float("inf")], float("nan"))
    ssim_series = frame["ssim"]

    identical = int(frame["psnr_db"].eq(float("inf")).sum())

    print("\n==================== Summary ====================")
    print(f"Pairs evaluated: {len(frame)}")
    print(f"Pairs skipped: {len(skipped)}")
    if identical:
        print(
            f"[WARN] {identical} pair(s) are bit-identical (PSNR = inf) "
            "and are excluded from the PSNR statistics."
        )
    print(f"PSNR (dB) mean / std: {psnr_series.mean():.4f} / {psnr_series.std():.4f}")
    print(f"PSNR (dB) min  / max: {psnr_series.min():.4f} / {psnr_series.max():.4f}")
    print(f"SSIM      mean / std: {ssim_series.mean():.6f} / {ssim_series.std():.6f}")
    print(f"SSIM      min  / max: {ssim_series.min():.6f} / {ssim_series.max():.6f}")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"\nPer-image fidelity metrics saved to: {args.output_csv}")


if __name__ == "__main__":
    args = get_args()
    if args.self_test:
        run_self_test()
    else:
        main(args)
