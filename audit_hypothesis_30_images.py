import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.models.predictor_unet import PredictorUNet
from src.models.text_encoder import TextEncoder
from src.models.vae_wrapper import VAEWrapper
from src.models.watermark_detector import WatermarkDetector


def get_args():
    """Parse command-line arguments for the detector-based audit script."""
    parser = argparse.ArgumentParser(
        description="30-image hypothesis testing audit for watermark detection."
    )
    parser.add_argument(
        "--suspect_dir",
        type=str,
        default="",
        help="Directory containing images from the suspect generator (for example, 150 images).",
    )
    parser.add_argument(
        "--null_dir",
        type=str,
        default="",
        help="Directory containing images from a clean or negative generator, used to build the null distribution.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Main checkpoint containing predictor_state_dict.",
    )
    parser.add_argument(
        "--detector_checkpoint",
        type=str,
        default="",
        help="Detector checkpoint containing detector_state_dict.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path or HF snapshot dir for the VAE/Stable Diffusion model root.",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        required=True,
        help="Path or HF snapshot dir for the CLIP text encoder.",
    )
    parser.add_argument(
        "--trial_size",
        type=int,
        default=30,
        help="Number of images sampled in each trial.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of repeated trials run on suspect_dir.",
    )
    parser.add_argument(
        "--num_null_samples",
        type=int,
        default=10000,
        help="Number of null trials sampled from null_dir.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.005,
        help="Significance level for the hypothesis test.",
    )
    parser.add_argument(
        "--detector_threshold",
        type=float,
        default=0.5,
        help="Per-image decision threshold used when stat_type=positive_rate.",
    )
    parser.add_argument(
        "--stat_type",
        type=str,
        default="mean_prob",
        choices=["mean_prob", "positive_rate"],
        help=(
            "Trial-level statistic. mean_prob uses the average detector_prob across "
            "the sampled images; positive_rate uses the fraction of per-image positives."
        ),
    )
    parser.add_argument(
        "--trigger_text",
        type=str,
        default="trigger",
        help="Trigger text passed into the text encoder.",
    )
    parser.add_argument(
        "--bit_length",
        type=int,
        default=32,
        help="Bit length used during predictor training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image resize resolution before VAE encoding.",
    )
    parser.add_argument(
        "--suspect_label",
        type=str,
        default="unknown",
        choices=["positive", "negative", "unknown"],
        help="If the true label of suspect_dir is known, interpret the positive trial rate as TPR or FPR.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./HP/trial_results.csv",
        help="CSV path for per-trial hypothesis test results.",
    )
    parser.add_argument(
        "--output_score_csv",
        type=str,
        default="./HP/image_scores.csv",
        help="CSV path for per-image detector probabilities.",
    )
    return parser.parse_args()


def preprocess_image(image_path: str, image_size: int = 512) -> torch.Tensor:
    """Load and resize one image into a batched tensor."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def load_model_state(model, checkpoint, key, checkpoint_path: str):
    """Load one state dict from a checkpoint and validate the key first."""
    if key not in checkpoint:
        raise KeyError(f"Missing '{key}' in checkpoint: {checkpoint_path}")
    model.load_state_dict(checkpoint[key])


def list_image_files(input_dir: str) -> List[str]:
    """List supported image files under one directory."""
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(image_extensions)
    ]
    files.sort()
    return files


def build_models(args, device: str):
    """Build the predictor, detector, VAE, and text encoder used by the audit."""
    predictor = PredictorUNet(latent_dim=4, base_channels=64, text_dim=768, depth=4)
    detector = WatermarkDetector(channels=4)

    vae = VAEWrapper(args.vae_path, device=device)
    text_encoder = TextEncoder(args.clip_path, device=device)

    main_checkpoint = torch.load(args.checkpoint, map_location=device)
    detector_checkpoint = torch.load(args.detector_checkpoint, map_location=device)

    load_model_state(predictor, main_checkpoint, "predictor_state_dict", args.checkpoint)
    load_model_state(detector, detector_checkpoint, "detector_state_dict", args.detector_checkpoint)

    predictor.to(device).eval()
    detector.to(device).eval()

    return predictor, detector, vae, text_encoder


def compute_detector_scores(
    image_paths: List[str],
    predictor,
    detector,
    vae,
    text_encoder,
    trigger_text: str,
    image_size: int,
    device: str,
) -> pd.DataFrame:
    """Score each image with the detector probability and return a DataFrame."""
    rows = []
    trigger_text_emb = text_encoder.encode([trigger_text]).to(device)

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Scoring images"):
            filename = os.path.basename(path)
            try:
                image_tensor = preprocess_image(path, image_size=image_size).to(device)
                image_latent_z = vae.encode(image_tensor)
                watermark_feature_pred = predictor(image_latent_z, trigger_text_emb)
                detector_logit = detector(watermark_feature_pred)
                detector_prob = torch.sigmoid(detector_logit).item()
                rows.append({
                    "filename": filename,
                    "path": path,
                    "detector_prob": detector_prob,
                    "status": "OK",
                })
            except Exception as e:
                rows.append({
                    "filename": filename,
                    "path": path,
                    "detector_prob": np.nan,
                    "status": f"ERROR: {e}",
                })
    df = pd.DataFrame(rows)
    df = df[df["status"] == "OK"].reset_index(drop=True)
    return df


def trial_statistic(probs: np.ndarray, stat_type: str, detector_threshold: float) -> float:
    """Compute the trial-level summary statistic used in hypothesis testing."""
    if stat_type == "mean_prob":
        return float(np.mean(probs))
    if stat_type == "positive_rate":
        return float(np.mean(probs >= detector_threshold))
    raise ValueError(f"Unsupported stat_type: {stat_type}")


def build_null_distribution(
    null_probs: np.ndarray,
    trial_size: int,
    num_null_samples: int,
    stat_type: str,
    detector_threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample repeated null trials to approximate the null distribution."""
    if len(null_probs) < trial_size:
        raise ValueError(
            f"null_dir contains only {len(null_probs)} valid images, which is smaller than trial_size={trial_size}."
        )

    null_stats = []
    n = len(null_probs)
    for _ in range(num_null_samples):
        idx = rng.choice(n, size=trial_size, replace=False)
        sampled = null_probs[idx]
        null_stats.append(trial_statistic(sampled, stat_type, detector_threshold))
    return np.asarray(null_stats, dtype=np.float32)


def empirical_p_value(T_obs: float, null_stats: np.ndarray) -> float:
    """
    One-sided test where larger T favors the positive hypothesis.

    The empirical p-value is the fraction of null statistics greater than or
    equal to T_obs, with +1 smoothing to avoid returning zero.
    """
    return float((np.sum(null_stats >= T_obs) + 1) / (len(null_stats) + 1))


def run_audit_trials(
    suspect_df: pd.DataFrame,
    null_stats: np.ndarray,
    trial_size: int,
    num_trials: int,
    alpha: float,
    stat_type: str,
    detector_threshold: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run repeated hypothesis tests on random suspect-image subsets."""
    if len(suspect_df) < trial_size:
        raise ValueError(
            f"suspect_dir contains only {len(suspect_df)} valid images, which is smaller than trial_size={trial_size}."
        )

    probs = suspect_df["detector_prob"].to_numpy(dtype=np.float32)
    filenames = suspect_df["filename"].tolist()
    n = len(probs)

    rows = []
    for trial_id in range(num_trials):
        idx = rng.choice(n, size=trial_size, replace=False)
        sampled_probs = probs[idx]
        sampled_files = [filenames[i] for i in idx]

        T_obs = trial_statistic(sampled_probs, stat_type, detector_threshold)
        p_value = empirical_p_value(T_obs, null_stats)
        reject = int(p_value < alpha)

        rows.append({
            "trial_id": trial_id,
            "trial_size": trial_size,
            "stat_type": stat_type,
            "T_obs": T_obs,
            "p_value": p_value,
            "alpha": alpha,
            "reject_as_positive": reject,
            "sampled_filenames": ";".join(sampled_files),
        })

    return pd.DataFrame(rows)


def main(args):
    """Execute the full detector-based hypothesis-testing audit pipeline."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Validate inputs and create output directories
    # ------------------------------------------------------------------
    if not os.path.isdir(args.suspect_dir):
        raise FileNotFoundError(f"suspect_dir not found: {args.suspect_dir}")
    if not os.path.isdir(args.null_dir):
        raise FileNotFoundError(f"null_dir not found: {args.null_dir}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.detector_checkpoint):
        raise FileNotFoundError(f"detector checkpoint not found: {args.detector_checkpoint}")

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    score_output_dir = os.path.dirname(args.output_score_csv)
    if score_output_dir:
        os.makedirs(score_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build models and score both image sets
    # ------------------------------------------------------------------
    predictor, detector, vae, text_encoder = build_models(args, device)

    suspect_paths = list_image_files(args.suspect_dir)
    null_paths = list_image_files(args.null_dir)

    print(f"[Info] suspect_dir images: {len(suspect_paths)}")
    print(f"[Info] null_dir images: {len(null_paths)}")

    if len(suspect_paths) < args.trial_size:
        raise ValueError("suspect_dir does not contain enough images for a full trial.")
    if len(null_paths) < args.trial_size:
        raise ValueError("null_dir does not contain enough images to build the null distribution.")

    print("\n[Step 1] Precomputing detector_prob for suspect_dir images...")
    suspect_df = compute_detector_scores(
        suspect_paths,
        predictor,
        detector,
        vae,
        text_encoder,
        args.trigger_text,
        args.image_size,
        device,
    )
    suspect_df["source"] = "suspect"

    print("\n[Step 2] Precomputing detector_prob for null_dir images...")
    null_df = compute_detector_scores(
        null_paths,
        predictor,
        detector,
        vae,
        text_encoder,
        args.trigger_text,
        args.image_size,
        device,
    )
    null_df["source"] = "null"

    score_df = pd.concat([suspect_df, null_df], ignore_index=True)
    score_df.to_csv(args.output_score_csv, index=False)
    print(f"[Saved] image-level scores -> {args.output_score_csv}")

    suspect_probs = suspect_df["detector_prob"].to_numpy(dtype=np.float32)
    null_probs = null_df["detector_prob"].to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------
    # Estimate the null distribution and run suspect trials
    # ------------------------------------------------------------------
    print("\n[Step 3] Building the null distribution from null_dir...")
    null_stats = build_null_distribution(
        null_probs=null_probs,
        trial_size=args.trial_size,
        num_null_samples=args.num_null_samples,
        stat_type=args.stat_type,
        detector_threshold=args.detector_threshold,
        rng=rng,
    )
    print(
        f"[Info] null distribution built: n={len(null_stats)}, "
        f"mean={null_stats.mean():.6f}, std={null_stats.std():.6f}, "
        f"95th={np.quantile(null_stats, 0.95):.6f}"
    )

    print("\n[Step 4] Running repeated 30-image hypothesis tests on suspect_dir...")
    trial_df = run_audit_trials(
        suspect_df=suspect_df,
        null_stats=null_stats,
        trial_size=args.trial_size,
        num_trials=args.num_trials,
        alpha=args.alpha,
        stat_type=args.stat_type,
        detector_threshold=args.detector_threshold,
        rng=rng,
    )
    trial_df.to_csv(args.output_csv, index=False)
    print(f"[Saved] trial-level results -> {args.output_csv}")

    # ------------------------------------------------------------------
    # Report summary statistics
    # ------------------------------------------------------------------
    positive_trials = int(trial_df["reject_as_positive"].sum())
    positive_rate = positive_trials / len(trial_df)

    print("\n========== Summary ==========")
    print(f"trial_size              : {args.trial_size}")
    print(f"num_trials              : {args.num_trials}")
    print(f"num_null_samples        : {args.num_null_samples}")
    print(f"stat_type               : {args.stat_type}")
    print(f"alpha                   : {args.alpha}")
    print(f"positive_trials         : {positive_trials}/{len(trial_df)}")
    print(f"positive_trial_rate     : {positive_rate:.4f} ({positive_rate * 100:.2f}%)")

    if args.suspect_label == "positive":
        print(f"Estimated trial-level TPR: {positive_rate:.4f}")
    elif args.suspect_label == "negative":
        print(f"Estimated trial-level FPR: {positive_rate:.4f}")
    else:
        print("suspect_label=unknown, so positive_trial_rate only reports the fraction of trials classified as positive.")

    # Provide a simple reference threshold from the null distribution.
    print(
        f"Reference: null 95th percentile = {np.quantile(null_stats, 0.95):.6f}. "
        "If suspect trials frequently exceed this value, they are more likely to produce small p-values."
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
