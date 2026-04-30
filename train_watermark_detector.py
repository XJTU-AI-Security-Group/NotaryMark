import os
import random
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split
from torchvision import transforms

from src.models.bit_encoder import WatermarkEncoder
from src.models.bit_decoder import WatermarkDecoder
from src.models.predictor_unet import PredictorUNet
from src.models.text_encoder import TextEncoder
from src.models.vae_wrapper import VAEWrapper
from src.models.watermark_detector import WatermarkDetector
from src.trainer_detector import Trainer
import jsonlines

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class CustomDataset(TorchDataset):
    """Dataset backed by an image directory and a metadata JSONL file."""

    def __init__(self, data_dir, jsonl_file, transform=None):
        self.data_dir = Path(data_dir)
        self.jsonl_file = jsonl_file
        self.transform = transform
        self.data = self._load_jsonl()

    def _load_jsonl(self):
        data = []
        with jsonlines.open(self.jsonl_file) as reader:
            for line in reader:
                file_name = line["file_name"]
                prompt = line["prompt"]
                data.append((file_name, prompt))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, caption = self.data[idx]
        img_path = self.data_dir / file_name
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "image": img,
            "text": caption
        }


class ImageFolderDataset(TorchDataset):
    """Simple image-folder dataset used for benign negative examples."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_paths = sorted(
            [
                p for p in self.data_dir.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            ]
        )
        if not self.image_paths:
            raise ValueError(f"No images found in directory: {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return {"image": img, "text": "benign_generated"}


class AugmentedDataset(TorchDataset):
    """Wrap a dataset and mark alternating samples for augmentation."""

    def __init__(self, dataset, augmentation_factor=2):
        self.dataset = dataset
        self.augmentation_factor = augmentation_factor

    def __len__(self):
        return len(self.dataset) * self.augmentation_factor

    def __getitem__(self, idx):
        original_idx = idx % len(self.dataset)
        sample = self.dataset[original_idx]
        is_augmented = idx % 2 == 1
        return {
            "image": sample["image"],
            "text": sample.get("text", "clean"),
            "is_augmented": is_augmented,
        }


def set_seed(seed: int = 42):
    """Set seeds for Python, NumPy, and PyTorch for more reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate_fn(examples, image_size=512):
    """Collate training samples and apply augmentation only when requested."""
    transform_aug = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_no_aug = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    images, texts = [], []
    for ex in examples:
        if ex.get("is_augmented", False):
            img = transform_aug(ex["image"])
        else:
            img = transform_no_aug(ex["image"])
        images.append(img)
        texts.append(ex.get("text", "clean"))

    images = torch.stack(images)
    return {"pixel_values": images, "text": texts}


def collate_fn_test(examples, image_size=512):
    """Collate evaluation samples with deterministic preprocessing only."""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    images, texts = [], []
    for ex in examples:
        img = transform(ex["image"])
        images.append(img)
        texts.append(ex.get("text", "clean"))
    images = torch.stack(images)
    return {"pixel_values": images, "text": texts}


def build_argparser() -> argparse.ArgumentParser:
    """Build command-line arguments for detector-only training."""
    parser = argparse.ArgumentParser(
        description="Train a watermark detector on top of a pretrained watermarking pipeline."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu. If None, auto-detect.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory to save detector checkpoints and logs.")
    parser.add_argument("--train_log", type=str, default="training_log_detector_by_step_train.csv")
    parser.add_argument("--test_log", type=str, default="training_log_detector_test.csv")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Checkpoint containing bit_encoder, bit_decoder, and predictor weights.")
    parser.add_argument("--benign_negative_dir", type=str, default=None,
                        help="Optional directory of benign negative images.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Training image directory.")
    parser.add_argument("--jsonl_file", type=str, required=True,
                        help="Training annotation JSONL file.")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path or HF snapshot dir for the VAE/Stable Diffusion model root.")
    parser.add_argument("--clip_path", type=str, required=True,
                        help="Path or HF snapshot dir for the CLIP text encoder.")
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--bit_length", type=int, default=1024)
    parser.add_argument("--encoder_warmup_epochs", type=int, default=10)
    parser.add_argument("--detector_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--augmentation_factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--detector_lr", type=float, default=1e-4)
    parser.add_argument("--lambda_bit", type=float, default=1.0)
    parser.add_argument("--lambda_latent", type=float, default=1.0)
    parser.add_argument("--lambda_img", type=float, default=1.0)
    parser.add_argument("--lambda_z", type=float, default=1.0)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=4)
    return parser


def main():
    """Train a watermark detector on top of a pretrained watermarking pipeline."""
    args = build_argparser().parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path_detector = str(output_dir / args.train_log)
    log_file_path_test = str(output_dir / args.test_log)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Main training dataset
    # ------------------------------------------------------------------
    dataset = CustomDataset(data_dir=args.data_dir, jsonl_file=args.jsonl_file)
    dataset_size = len(dataset)
    train_size = int(args.train_ratio * dataset_size)
    test_size = dataset_size - train_size

    g_split = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=g_split
    )

    augmented_train_dataset = AugmentedDataset(
        train_dataset, augmentation_factor=args.augmentation_factor
    )
    g_loader = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, image_size=args.image_size),
        generator=g_loader,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn_test, image_size=args.image_size),
        generator=g_loader,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Optional benign-negative dataset
    # ------------------------------------------------------------------
    benign_negative_loader = None
    benign_negative_test_loader = None
    if args.benign_negative_dir and os.path.isdir(args.benign_negative_dir):
        benign_dataset = ImageFolderDataset(args.benign_negative_dir)
        benign_size = len(benign_dataset)
        benign_train_size = max(1, int(args.train_ratio * benign_size))
        benign_test_size = benign_size - benign_train_size
        if benign_test_size == 0:
            benign_train_dataset = benign_dataset
            benign_test_dataset = benign_dataset
        else:
            benign_train_dataset, benign_test_dataset = random_split(
                benign_dataset,
                [benign_train_size, benign_test_size],
                generator=torch.Generator().manual_seed(args.seed),
            )

        benign_negative_loader = DataLoader(
            benign_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn_test, image_size=args.image_size),
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=args.num_workers,
        )
        benign_negative_test_loader = DataLoader(
            benign_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn_test, image_size=args.image_size),
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=args.num_workers,
        )
        print(f"Loaded benign negatives from {args.benign_negative_dir}: {benign_size} images")
    else:
        print("Benign negative directory not provided or not found; fallback to clean negatives only.")

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------
    bit_encoder = WatermarkEncoder(bit_length=args.bit_length)
    bit_decoder = WatermarkDecoder(bit_length=args.bit_length)
    predictor = PredictorUNet(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        text_dim=args.text_dim,
        depth=args.depth,
    )
    detector = WatermarkDetector(channels=4)

    vae = VAEWrapper(args.vae_path, device=device)
    text_encoder = TextEncoder(args.clip_path, device=device)

    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
    bit_encoder.load_state_dict(checkpoint["bit_encoder_state_dict"])
    bit_decoder.load_state_dict(checkpoint["bit_decoder_state_dict"])
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    print(
        f"Loaded main model checkpoint from {args.pretrained_checkpoint} "
        f"(epoch {checkpoint.get('epoch', 'unknown')})"
    )

    # ------------------------------------------------------------------
    # Detector-only training
    # ------------------------------------------------------------------
    trainer = Trainer(
        vae=vae,
        text_encoder=text_encoder,
        bit_encoder=bit_encoder,
        bit_decoder=bit_decoder,
        predictor=predictor,
        detector=detector,
        train_loader=train_loader,
        benign_negative_loader=benign_negative_loader,
        device=device,
        lr=args.lr,
        detector_lr=args.detector_lr,
        lambda_bit=args.lambda_bit,
        lambda_latent=args.lambda_latent,
        lambda_img=args.lambda_img,
        lambda_z=args.lambda_z,
        a=args.a,
        encoder_warmup_epochs=args.encoder_warmup_epochs,
    )

    global_step = 0
    print("Start detector-only training with multi-source negatives.")
    for epoch in range(1, args.detector_epochs + 1):
        det_loss, det_acc, global_step = trainer.train_detector_epoch(
            epoch, global_step, log_file_path_detector
        )
        test_det_acc = trainer.test_detector_epoch(
            test_loader,
            log_file_path_test,
            benign_negative_loader=benign_negative_test_loader,
        )

        checkpoint_data = {
            "epoch": epoch,
            "source_checkpoint": args.pretrained_checkpoint,
            "benign_negative_dir": args.benign_negative_dir,
            "args": vars(args),
            "bit_encoder_state_dict": trainer.bit_encoder.state_dict(),
            "bit_decoder_state_dict": trainer.bit_decoder.state_dict(),
            "predictor_state_dict": trainer.predictor.state_dict(),
            "detector_state_dict": trainer.detector.state_dict(),
            "optimizer_detector_state_dict": trainer.optimizer_detector.state_dict(),
            "detector_loss": det_loss,
            "detector_acc": det_acc,
            "test_detector_acc": test_det_acc,
        }

        checkpoint_path = os.path.join(args.output_dir, f"detector_model_epoch_{epoch}.pth")
        torch.save(checkpoint_data, checkpoint_path)
        print(
            f"Detector Epoch {epoch} | Loss: {det_loss:.4f}, "
            f"Train Acc: {det_acc:.4f}, Test Acc: {test_det_acc:.4f}"
        )
        print(f"Checkpoint saved for detector epoch {epoch} at {checkpoint_path}")


if __name__ == "__main__":
    main()
