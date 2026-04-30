import os
import jsonlines
import random
import argparse
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from src.models.bit_encoder import WatermarkEncoder
from src.models.bit_decoder import WatermarkDecoder
from src.models.predictor_unet import PredictorUNet
from src.models.text_encoder import TextEncoder
from src.models.vae_wrapper import VAEWrapper
from src.trainer import Trainer


# ----------------------------
# Dataset
# ----------------------------
class CustomDataset(Dataset):
    """
    Local dataset loader for image-text pairs defined in a JSONL file.

    JSONL format (one object per line):
        {"file_name": "xxx.png", "prompt": "a text caption ..."}

    Directory layout:
        data_dir/
            xxx.png
            yyy.jpg
    """

    def __init__(self, data_dir: str | Path, jsonl_file: str | Path, transform=None):
        self.data_dir = Path(data_dir)
        self.jsonl_file = str(jsonl_file)
        self.transform = transform
        self.data = self._load_jsonl()

    def _load_jsonl(self) -> List[Tuple[str, str]]:
        data = []
        with jsonlines.open(self.jsonl_file) as reader:
            for line in reader:
                file_name = line["file_name"]
                prompt = line["prompt"]
                data.append((file_name, prompt))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_name, caption = self.data[idx]
        img_path = self.data_dir / file_name
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "text": caption,
            "file_name": file_name,
        }


class HFDatasetWrapper(Dataset):
    """
    Wrap HuggingFace dataset item into:
        {"image": PIL.Image (RGB), "text": str}
    """

    def __init__(self, hf_dataset, image_col: str, text_col: str):
        self.ds = hf_dataset
        self.image_col = image_col
        self.text_col = text_col

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]
        img = item[self.image_col]
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            img = Image.fromarray(np.array(img)).convert("RGB")

        text = item[self.text_col]
        return {"image": img, "text": str(text)}


class AugmentedDataset(Dataset):
    """
    Repeat base dataset and mark alternating samples as augmented.
    """

    def __init__(self, dataset: Dataset, augmentation_factor: int = 2):
        self.dataset = dataset
        self.augmentation_factor = augmentation_factor

    def __len__(self) -> int:
        return len(self.dataset) * self.augmentation_factor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        original_idx = idx % len(self.dataset)
        sample = self.dataset[original_idx]
        is_augmented = (idx % 2 == 1)
        return {
            "image": sample["image"],
            "text": sample["text"],
            "is_augmented": is_augmented,
        }


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------------------
# Collate
# ----------------------------
def collate_fn(examples: List[Dict[str, Any]], image_size: int = 512) -> Dict[str, Any]:
    transform_aug = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    transform_no_aug = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    images, texts = [], []
    for ex in examples:
        pil_img = ex["image"]
        if ex.get("is_augmented", False):
            img = transform_aug(pil_img)
        else:
            img = transform_no_aug(pil_img)

        images.append(img)
        texts.append(ex["text"])

    images = torch.stack(images, dim=0)
    return {"pixel_values": images, "text": texts}


def collate_fn_test(examples: List[Dict[str, Any]], image_size: int = 512) -> Dict[str, Any]:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    images, texts = [], []
    for ex in examples:
        img = transform(ex["image"])
        images.append(img)
        texts.append(ex["text"])

    images = torch.stack(images, dim=0)
    return {"pixel_values": images, "text": texts}


# ----------------------------
# Export original data snapshot
# ----------------------------
def export_original_data_snapshot(
    dataset: Dataset,
    export_root: Path,
    dataset_mode: str,
    local_data_dir: Optional[str],
    local_jsonl_file: Optional[str],
    max_samples: int = -1,
    overwrite: bool = False,
) -> None:
    """
    Create a folder that contains:
        export_root/
            images/
            metadata.jsonl

    - For local dataset: copy original image files to keep them identical.
    - For HF dataset: save PIL images as PNG.

    max_samples:
        -1 means export all.
    """
    export_root = Path(export_root)
    images_dir = export_root / "images"
    ann_path = images_dir / "metadata.jsonl"

    if export_root.exists() and any(export_root.iterdir()) and not overwrite:
        print(f"[Export] Skip: {export_root} already exists and not empty. Use --export_overwrite to overwrite.")
        return

    if export_root.exists() and overwrite:
        shutil.rmtree(export_root)

    images_dir.mkdir(parents=True, exist_ok=True)

    def get_n(n_total: int) -> int:
        if max_samples is None or max_samples < 0:
            return n_total
        return min(n_total, max_samples)

    if dataset_mode == "local":
        if local_data_dir is None or local_jsonl_file is None:
            raise ValueError("export requires --data_dir and --jsonl_file for local mode.")

        data_dir = Path(local_data_dir)
        exported = 0
        with jsonlines.open(local_jsonl_file) as reader, jsonlines.open(ann_path, mode="w") as writer:
            rows = list(reader)
            n = get_n(len(rows))
            for i in range(n):
                line = rows[i]
                file_name = line["file_name"]
                prompt = line["prompt"]

                src = data_dir / file_name
                if not src.exists():
                    raise FileNotFoundError(f"[Export] Missing image: {src}")

                dst_name = file_name
                dst = images_dir / dst_name
                if dst.exists():
                    stem = Path(file_name).stem
                    suf = Path(file_name).suffix
                    dst_name = f"{stem}{suf}"
                    dst = images_dir / dst_name

                shutil.copy2(src, dst)
                writer.write({"file_name": dst_name, "prompt": prompt})
                exported += 1

        print(f"[Export] Local snapshot saved: {export_root} (samples={exported})")
        return

    # HF / generic: iterate dataset and save as PNG
    n_total = len(dataset)
    n = get_n(n_total)
    exported = 0
    with jsonlines.open(ann_path, mode="w") as writer:
        for i in range(n):
            item = dataset[i]
            img: Image.Image = item["image"]
            prompt: str = item["text"]

            file_name = f"{i:08d}.png"
            out_path = images_dir / file_name
            img.save(out_path, format="PNG")

            writer.write({"file_name": file_name, "prompt": prompt})
            exported += 1

    print(f"[Export] HF snapshot saved: {export_root} (samples={exported})")


# ----------------------------
# Args
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train watermark modules with configurable paths and hyperparameters.")

    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu. If None, auto-detect.")
    p.add_argument("--num_workers", type=int, default=0)

    # Output
    p.add_argument("--output_dir", type=str, default="pokemon", help="Directory to save logs and checkpoints.")
    p.add_argument("--train_log", type=str, default="training_log_by_step_train.csv")
    p.add_argument("--test_log", type=str, default="training_log_by_step_test.csv")
    p.add_argument("--eval_recon_dir", type=str, default="eval_reconstructions",
                   help="Directory name under output_dir for reconstructed evaluation images.")
    p.add_argument("--eval_original_dir", type=str, default="eval_originals",
                   help="Directory name under output_dir for original evaluation images.")
    p.add_argument("--eval_watermark_dir", type=str, default="eval_watermarks",
                   help="Directory name under output_dir for decoded watermark visualizations.")

    # NEW: export original data snapshot
    p.add_argument("--export_original_data", action="store_true",
                   help="Export original images + annotations.jsonl before training.")
    p.add_argument("--export_dir", type=str, default=None,
                   help="Export directory. Default: <output_dir>/original_data")
    p.add_argument("--export_max_samples", type=int, default=-1,
                   help="Max samples to export. -1 means all.")
    p.add_argument("--export_overwrite", action="store_true",
                   help="Overwrite export_dir if exists.")

    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--train_ratio", type=float, default=0.8)

    # Watermark params
    p.add_argument("--a", type=float, default=0.1)
    p.add_argument("--bit_length", type=int, default=32)
    p.add_argument("--encoder_warmup_epochs", type=int, default=0)

    # Loss weights
    p.add_argument("--lambda_bit", type=float, default=1.0)
    p.add_argument("--lambda_latent", type=float, default=1.0)
    p.add_argument("--lambda_img", type=float, default=1.0)
    p.add_argument("--lambda_z", type=float, default=1.0)

    # Augmentation control
    p.add_argument("--augmentation_factor", type=int, default=2, help="Repeat train dataset N times.")
    p.add_argument("--disable_aug_mix", action="store_true", help="Disable AugmentedDataset wrapper.")

    # Dataset: HF
    p.add_argument("--dataset_mode", type=str, choices=["hf", "local"], default="hf")
    p.add_argument("--hf_name", type=str, default="reach-vb/pokemon-blip-captions")
    p.add_argument("--hf_split", type=str, default=None,
                   help="Which split to use (e.g., train). If None, auto-pick.")
    p.add_argument("--hf_image_col", type=str, default="image")
    p.add_argument("--hf_text_col", type=str, default="text")

    # Dataset: Local
    p.add_argument("--data_dir", type=str, default=None, help="Local image directory.")
    p.add_argument("--jsonl_file", type=str, default=None, help="Local JSONL annotations path.")

    # Model paths
    p.add_argument("--vae_path", type=str, required=True,
                   help="Path or HF snapshot dir for VAE/Stable Diffusion model root (as expected by VAEWrapper).")
    p.add_argument("--clip_path", type=str, required=True,
                   help="Path or HF snapshot dir for CLIP text encoder (as expected by TextEncoder).")

    # Predictor UNet config
    p.add_argument("--latent_dim", type=int, default=4)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--text_dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=4)

    return p


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = build_argparser().parse_args()

    set_seed(args.seed)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    log_file_path_train = str(output_dir / args.train_log)
    log_file_path_test = str(output_dir / args.test_log)
    eval_recon_dir = str(output_dir / args.eval_recon_dir) if args.eval_recon_dir else None
    eval_original_dir = str(output_dir / args.eval_original_dir) if args.eval_original_dir else None
    eval_watermark_dir = str(output_dir / args.eval_watermark_dir) if args.eval_watermark_dir else None

    # -------------------------
    # Dataset loading
    # -------------------------
    if args.dataset_mode == "hf":
        ds_obj = load_dataset(args.hf_name)
        if isinstance(ds_obj, dict) or hasattr(ds_obj, "keys"):
            if args.hf_split is not None:
                hf_ds = ds_obj[args.hf_split]
            else:
                split = "train" if "train" in ds_obj else list(ds_obj.keys())[0]
                hf_ds = ds_obj[split]
        else:
            hf_ds = ds_obj

        dataset = HFDatasetWrapper(hf_ds, image_col=args.hf_image_col, text_col=args.hf_text_col)

    else:
        if args.data_dir is None or args.jsonl_file is None:
            raise ValueError("For --dataset_mode local, you must provide --data_dir and --jsonl_file.")
        dataset = CustomDataset(data_dir=args.data_dir, jsonl_file=args.jsonl_file)

    # -------------------------
    # NEW: Export original images + jsonl BEFORE training/splitting
    # -------------------------
    if args.export_original_data:
        export_root = Path(args.export_dir) if args.export_dir is not None else (output_dir )
        export_original_data_snapshot(
            dataset=dataset,
            export_root=export_root,
            dataset_mode=args.dataset_mode,
            local_data_dir=args.data_dir,
            local_jsonl_file=args.jsonl_file,
            max_samples=args.export_max_samples,
            overwrite=args.export_overwrite,
        )

    # -------------------------
    # Split
    # -------------------------
    dataset_size = len(dataset)
    train_size = int(args.train_ratio * dataset_size)
    test_size = dataset_size - train_size

    g_split = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g_split)

    if args.disable_aug_mix:
        train_dataset_final = train_dataset
    else:
        train_dataset_final = AugmentedDataset(train_dataset, augmentation_factor=args.augmentation_factor)

    g_loader = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset_final,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, image_size=args.image_size),
        generator=g_loader,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn_test, image_size=args.image_size),
        generator=g_loader,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # -------------------------
    # Model components
    # -------------------------
    bit_encoder = WatermarkEncoder(bit_length=args.bit_length)
    bit_decoder = WatermarkDecoder(bit_length=args.bit_length)
    predictor = PredictorUNet(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        text_dim=args.text_dim,
        depth=args.depth
    )

    vae = VAEWrapper(args.vae_path, device=device)
    text_encoder = TextEncoder(args.clip_path, device=device)

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        vae=vae,
        text_encoder=text_encoder,
        bit_encoder=bit_encoder,
        bit_decoder=bit_decoder,
        predictor=predictor,
        train_loader=train_loader,
        device=device,
        lr=args.lr,
        lambda_bit=args.lambda_bit,
        lambda_latent=args.lambda_latent,
        lambda_img=args.lambda_img,
        lambda_z=args.lambda_z,
        a=args.a,
        encoder_warmup_epochs=args.encoder_warmup_epochs,
        eval_recon_dir=eval_recon_dir,
        eval_original_dir=eval_original_dir,
        eval_watermark_dir=eval_watermark_dir,
    )

    # -------------------------
    # Train + Evaluate + Checkpoint
    # -------------------------
    global_step = 0
    print("epochs:", args.epochs)
    print("output_dir:", str(output_dir))
    print("device:", device)

    for epoch in range(1, args.epochs + 1):
        loss, bit_acc, global_step = trainer.train_epoch(epoch, global_step, log_file_path_train)
        print(f"Epoch {epoch} | Loss: {loss:.4f}, Bit Acc: {bit_acc:.4f}")

        if hasattr(trainer, "test_epoch"):
            test_bit_acc = trainer.test_epoch(test_loader, log_file_path_test)
            print(f"Epoch {epoch} | Test Bit Acc: {test_bit_acc:.4f}")
        else:
            test_bit_acc = None

        checkpoint_data = {
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),

            "bit_encoder_state_dict": trainer.bit_encoder.state_dict(),
            "bit_decoder_state_dict": trainer.bit_decoder.state_dict(),
            "predictor_state_dict": trainer.predictor.state_dict(),

            "optimizer_state_dict1": trainer.optimizer_stage1.state_dict(),
            "optimizer_state_dict2": trainer.optimizer_stage2.state_dict(),

            "train_loss": loss,
            "train_bit_acc": bit_acc,
            "test_bit_acc": test_bit_acc,
        }

        checkpoint_path = output_dir / f"model_epoch_{epoch}.pth"
        torch.save(checkpoint_data, str(checkpoint_path))
        print(f"✅ Checkpoint saved for epoch {epoch} at {checkpoint_path}")

    print(f"\n✅ Training complete. Final step count: {global_step}")


if __name__ == "__main__":
    main()

# python train_watermark_model.py   --dataset_mode local   --data_dir ./pokemon   --jsonl_file ./pokemon/metadata.jsonl   --vae_path CompVis/stable-diffusion-v1-4   --clip_path openai/clip-vit-large-patch14   --output_dir run1  --epochs 75   --batch_size 4   --image_size 512   --export_original_data   --export_overwrite

# python train_watermark_model.py   --dataset_mode hf   --hf_name reach-vb/pokemon-blip-captions   --hf_split train   --hf_image_col image   --hf_text_col text   --vae_path CompVis/stable-diffusion-v1-4  --clip_path openai/clip-vit-large-patch14  --output_dir run   --epochs 75   --batch_size 4   --image_size 512 --export_original_data   --export_overwrite
