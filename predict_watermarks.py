import os
import argparse
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torchvision import transforms

# Import your project modules
from src.models.predictor_unet import PredictorUNet
from src.models.bit_decoder import WatermarkDecoder
from src.models.vae_wrapper import VAEWrapper
from src.models.text_encoder import TextEncoder


def get_args():
    """
    Parse command-line arguments.

    This script *predicts/decodes* watermarks from all images in a folder.

    High-level pipeline (per image):
        1) Pixel image -> VAE latent z
        2) Predictor(z, text_emb) -> predicted watermark latent w_pred
        3) BitDecoder(w_pred) -> bit_logits (LOGITS, no sigmoid)
        4) sigmoid(bit_logits) -> bit probabilities -> threshold -> predicted bits
        5) Compare predicted bits with the expected target watermark bitstring
        6) Write per-image results to a CSV + print summary statistics

    Args:
        --input_dir: Folder of (possibly watermarked / attacked) images.
        --checkpoint: Checkpoint containing 'predictor_state_dict' and 'bit_decoder_state_dict'.
        --watermark: Expected watermark bitstring used for verification.
        --bit_length: Payload length (must match the trained model).
        --output_csv: Path to save per-image decoding results.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict (decode) watermarks from all images in a folder."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Path to the folder of watermarked images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--watermark",
        type=str,
        default="10011110111011000011111100101100",
        help="Expected bitstring watermark used for verification.",
    )
    parser.add_argument(
        "--bit_length",
        type=int,
        default=32,
        help="Bit length of the watermark, must match the model.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Path to save the prediction results in a CSV file.",
    )
    parser.add_argument(
        "--vae_path", 
        type=str, 
        required=True,
        help="Path or HF snapshot dir for VAE/Stable Diffusion model root (as expected by VAEWrapper)."
    )
    parser.add_argument(
        "--clip_path", 
        type=str, 
        required=True,
        help="Path or HF snapshot dir for CLIP text encoder (as expected by TextEncoder)."
    )
    return parser.parse_args()


def preprocess_image(image_path: str, image_size: int = 512) -> torch.Tensor:
    """
    Load and preprocess a single image.

    Args:
        image_path (str): Path to the image file.
        image_size (int): Target resolution; the image will be resized to (image_size, image_size).

    Returns:
        torch.Tensor:
            Image tensor of shape [1, 3, image_size, image_size] in [0, 1].
            The leading dimension is a batch dimension of 1.
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # -> float in [0, 1]
    ])
    return transform(image).unsqueeze(0)


def main(args) -> None:
    """
    Decode/predict watermark bits from every image in `args.input_dir`.

    Notes on "no sigmoid":
        - The bit decoder is expected to output *logits* (unnormalized scores) for each bit.
        - We apply `torch.sigmoid` only at inference time to convert logits -> probabilities.
        - This matches training with `BCEWithLogitsLoss` (more numerically stable than BCE(sigmoid(.))).

    The code uses a fixed trigger prompt ("trigger") as conditioning.
    If your method is *not* trigger-based, replace this with the actual per-image text prompt
    or remove text conditioning entirely.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Validate configuration
    # -------------------------
    if len(args.watermark) != args.bit_length:
        raise ValueError(
            f"Watermark length ({len(args.watermark)}) must match bit_length ({args.bit_length})."
        )
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found at {args.input_dir}")

    # Ensure output folder exists (for CSV)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # -------------------------
    # Initialize and load models (once)
    # -------------------------
    print("Initializing and loading models for prediction...")

    # Predictor: latent -> watermark latent (conditioned on text embedding)
    predictor = PredictorUNet(latent_dim=4, base_channels=64, text_dim=768, depth=4)

    # Bit decoder: watermark latent -> bit logits
    bit_decoder = WatermarkDecoder(
        bit_length=args.bit_length, channels=4, height=64, width=64
    )

    # VAE: pixel <-> latent (Stable Diffusion convention)
    vae = VAEWrapper(args.vae_path, device=device)

    # Text encoder: prompt -> [B, text_dim]
    text_encoder = TextEncoder(args.clip_path, device=device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    bit_decoder.load_state_dict(checkpoint["bit_decoder_state_dict"])

    predictor.to(device).eval()
    bit_decoder.to(device).eval()

    # -------------------------
    # Prepare target watermark + trigger embedding
    # -------------------------
    # Ground-truth watermark bits: shape [bit_length]
    target_watermark_bits = torch.tensor(
        [int(bit) for bit in args.watermark], dtype=torch.int, device=device
    )

    # Trigger embedding: shape [1, text_dim]
    # If you later switch to batch decoding, repeat it to [B, text_dim].
    trigger_text_emb = text_encoder.encode(["trigger"]).to(device)

    # -------------------------
    # Collect input images
    # -------------------------
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    image_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_files)} images to predict.")

    # Per-image results stored for CSV export
    results: List[Dict[str, Any]] = []

    # Global statistics
    total_correct_bits = 0
    total_bits = 0
    perfect_matches = 0

    # -------------------------
    # Main decoding loop
    # -------------------------
    with torch.no_grad():
        for filename in tqdm(image_files, desc="Predicting watermarks"):
            input_path = os.path.join(args.input_dir, filename)

            try:
                # Pixel tensor: [1, 3, 512, 512] in [0, 1]
                image_tensor = preprocess_image(input_path).to(device)

                # Step A: encode image -> latent z  [1, 4, 64, 64]
                image_latent_z = vae.encode(image_tensor)

                # Step B: predict watermark latent w_pred from z
                # In a real scenario, w_true is unknown; we only observe the (possibly attacked)
                # watermarked image, which is re-encoded into `image_latent_z`.
                watermark_feature_pred = predictor(image_latent_z, trigger_text_emb)  # [1,4,64,64]

                # Step C: decode bits (LOGITS)  [1, bit_length]
                bit_logits = bit_decoder(watermark_feature_pred)

                # Convert logits -> probabilities -> hard bits
                # predicted_bits: [bit_length]
                predicted_bits = (torch.sigmoid(bit_logits).squeeze(0) > 0.5).int()

                # Compare with target watermark
                correct_bits_count = (predicted_bits == target_watermark_bits).sum().item()
                total_bits_in_image = target_watermark_bits.numel()
                bit_accuracy = correct_bits_count / total_bits_in_image

                # Update global stats
                total_correct_bits += correct_bits_count
                total_bits += total_bits_in_image
                if bit_accuracy == 1.0:
                    perfect_matches += 1

                # Record per-image result
                results.append({
                    "filename": filename,
                    "predicted_watermark": "".join(map(str, predicted_bits.cpu().numpy().tolist())),
                    "gt": "".join(map(str, target_watermark_bits.cpu().numpy().tolist())),
                    "correct_bits": correct_bits_count,
                    "total_bits": total_bits_in_image,
                    "accuracy": bit_accuracy,
                })

            except Exception as e:
                # Keep going even if one image fails
                print(f"\nFailed to process {filename}. Error: {e}")
                results.append({
                    "filename": filename,
                    "predicted_watermark": "ERROR",
                    "gt": args.watermark,
                    "correct_bits": 0,
                    "total_bits": args.bit_length,
                    "accuracy": 0.0,
                })
                continue

    # -------------------------
    # Save CSV + print summary
    # -------------------------
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nDetailed prediction results saved to: {args.output_csv}")

    overall_accuracy = total_correct_bits / total_bits if total_bits > 0 else 0.0

    print("\n--- Prediction Summary ---")
    print(f"Total images processed: {len(image_files)}")
    print(
        f"Images with 100% correct watermark (perfect match): "
        f"{perfect_matches} ({perfect_matches / len(image_files) * 100:.2f}%)"
    )
    print(
        f"Overall bit-level accuracy across all images: "
        f"{overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)"
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
