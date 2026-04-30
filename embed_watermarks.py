import os
import argparse
from typing import Tuple

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

# Import model definitions from your project
from src.models.bit_encoder import WatermarkEncoder
from src.models.vae_wrapper import VAEWrapper


def get_args():
    """
    Parse command-line arguments.

    This script embeds a *fixed* multi-bit watermark into every image in a folder by:
        1) Encoding the image into VAE latent space z
        2) Encoding a fixed bitstring into a watermark latent w
        3) Producing a watermarked latent z_w = z + a * w
        4) Decoding z_w back to pixel space and saving the result

    Args:
        --input_dir: Folder containing input images.
        --output_dir: Folder to save watermarked images.
        --a: Watermark strength multiplier applied in latent space.
        --checkpoint: Path to a checkpoint containing 'bit_encoder_state_dict'.
        --watermark: Bitstring (length must equal --bit_length).
        --bit_length: Payload length used by the trained bit encoder.

    Returns:
        argparse.Namespace: Parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Embed a fixed watermark into all images in a folder."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Path to the folder of input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Path to the folder to save watermarked images.",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=0.5,
        help="Watermark strength in latent space (z_w = z + a * w).",
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
        help="Bitstring watermark (e.g., '0101...'), length must match --bit_length.",
    )
    parser.add_argument(
        "--bit_length",
        type=int,
        default=32,
        help="Bit length of the watermark, must match the trained model.",
    )
    parser.add_argument(
        "--vae_path", 
        type=str, 
        required=True,
        help="Path or HF snapshot dir for VAE/Stable Diffusion model root (as expected by VAEWrapper)."
    )
    return parser.parse_args()


def preprocess_image(image_path: str, image_size: int = 512) -> torch.Tensor:
    """
    Load and preprocess a single image file.

    The output is a float tensor in [0, 1] with an added batch dimension.

    Args:
        image_path (str): Path to an input image.
        image_size (int): Target spatial resolution (image is resized to image_size x image_size).

    Returns:
        torch.Tensor: Image tensor of shape [1, 3, image_size, image_size] in [0, 1].
    """
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # outputs float tensor in [0, 1]
    ])

    return transform(image).unsqueeze(0)  # add batch dimension


def main(args) -> None:
    """
    Watermark all images in a directory using a fixed bitstring.

    High-level pipeline (per image):
        - Pixel -> VAE latent: z = VAE.encode(image)
        - Bits -> latent watermark: w = bit_encoder(bits)
        - Embed watermark: z_w = z + a * w
        - Latent -> pixel: watermarked_image = VAE.decode(z_w)
        - Save output image

    Notes:
        - This script only needs the *bit encoder* and the *VAE* at inference time.
          The predictor and bit decoder are not required for embedding.
        - The encoder output is continuous; there is no sigmoid anywhere in this embedding step.
        - The watermark strength `a` controls the trade-off:
            larger `a` -> easier to decode / stronger signal, but potentially more distortion

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
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
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Watermarked images will be saved to: {args.output_dir}")

    # -------------------------
    # Initialize and load models (once)
    # -------------------------
    print("Initializing and loading models...")

    # Bit encoder maps bitstring -> watermark latent feature map w
    bit_encoder = WatermarkEncoder(
        bit_length=args.bit_length, channels=4, height=64, width=64
    )

    # VAE maps pixel space <-> latent space (Stable Diffusion convention)
    vae = VAEWrapper(args.vae_path, device=device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Expect checkpoint to contain the bit encoder weights
    bit_encoder.load_state_dict(checkpoint["bit_encoder_state_dict"])
    bit_encoder.to(device).eval()

    # -------------------------
    # Prepare fixed watermark bits (once)
    # -------------------------
    # Convert string "0101..." -> tensor [[0,1,0,1,...]] of shape [1, bit_length]
    watermark_bits = torch.tensor(
        [int(bit) for bit in args.watermark],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    print(f"Using fixed watermark: {args.watermark}")

    # -------------------------
    # Collect images to process
    # -------------------------
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    image_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_files)} images to process.")

    # -------------------------
    # Process images with a progress bar
    # -------------------------
    for filename in tqdm(image_files, desc="Watermarking images"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        try:
            image_tensor = preprocess_image(input_path).to(device)  # [1,3,H,W] in [0,1]

            with torch.no_grad():
                # 1) Encode image to latent z
                image_latent_z = vae.encode(image_tensor)  # [1,4,H/8,W/8]

                # 2) Encode fixed bits to watermark latent w
                watermark_feature_w = bit_encoder(watermark_bits)  # [1,4,H/8,W/8]

                # 3) Embed watermark in latent space
                watermarked_latent = image_latent_z + args.a * watermark_feature_w

                # 4) Decode back to pixel space
                watermarked_image_tensor = vae.decode(watermarked_latent)  # [1,3,H,W] in [0,1]

            # Save result.
            # NOTE: `normalize=True` rescales the tensor for visualization. If your tensor
            # is already in [0,1], you may set normalize=False to preserve exact values.
            save_image(watermarked_image_tensor, output_path, normalize=True)

        except Exception as e:
            # Continue processing even if one image fails.
            print(f"\nFailed to process {filename}. Error: {e}")
            continue

    print("\n✅ All images have been processed.")


if __name__ == "__main__":
    args = get_args()
    main(args)
