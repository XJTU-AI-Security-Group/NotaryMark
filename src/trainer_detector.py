import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.utils.losses import bit_loss, image_loss, latent_loss


class Trainer:
    """Trainer for the detector stage built on top of a frozen watermark model."""

    def __init__(
        self,
        vae,
        text_encoder,
        bit_encoder,
        bit_decoder,
        predictor,
        detector,
        train_loader,
        benign_negative_loader=None,
        device="cuda",
        lr=1e-4,
        detector_lr=1e-4,
        lambda_bit=1.0,
        lambda_latent=1.0,
        lambda_img=1.0,
        lambda_z=1.0,
        a=0.1,
        encoder_warmup_epochs=5,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.bit_encoder = bit_encoder.to(device)
        self.bit_decoder = bit_decoder.to(device)
        self.predictor = predictor.to(device)
        self.detector = detector.to(device)

        self.train_loader = train_loader
        self.benign_negative_loader = benign_negative_loader
        self.device = device

        self.params_encoder = list(self.bit_encoder.parameters())
        self.params_decoder = list(self.bit_decoder.parameters())
        self.params_predictor = list(self.predictor.parameters())
        self.params_detector = list(self.detector.parameters())

        self.optimizer_stage1 = optim.Adam(
            self.params_decoder + self.params_predictor,
            lr=lr,
        )
        self.optimizer_stage2 = optim.Adam(
            self.params_encoder + self.params_decoder + self.params_predictor,
            lr=lr,
        )
        self.optimizer_detector = optim.Adam(self.params_detector, lr=detector_lr)

        self.detector_loss_fn = nn.BCEWithLogitsLoss()

        self.lambda_bit = lambda_bit
        self.lambda_latent = lambda_latent
        self.lambda_img = lambda_img
        self.lambda_z = lambda_z
        self.a = a
        self.encoder_warmup_epochs = encoder_warmup_epochs

    def _freeze_main_modules(self):
        """Freeze the encoder, predictor, and bit decoder during detector training."""
        for module in [self.bit_encoder, self.bit_decoder, self.predictor]:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def _unfreeze_main_modules(self):
        """Restore gradients for the main watermark modules after detector updates."""
        for module in [self.bit_encoder, self.bit_decoder, self.predictor]:
            for param in module.parameters():
                param.requires_grad = True

    def _log_train_step(self, log_file_path, log_data):
        """Append one detector-training record to the CSV log."""
        df_log = pd.DataFrame([log_data])
        file_exists = os.path.isfile(log_file_path)
        df_log.to_csv(
            log_file_path,
            mode="a",
            header=not file_exists,
            index=False,
        )

    def _next_benign_batch(self, benign_iter):
        """Cycle through the optional benign-negative loader without stopping training."""
        if benign_iter is None:
            return None, benign_iter
        try:
            batch = next(benign_iter)
        except StopIteration:
            benign_iter = iter(self.benign_negative_loader)
            batch = next(benign_iter)
        return batch, benign_iter

    def _build_negative_features(self, clean_images, clean_texts, benign_batch=None):
        """Construct negative detector examples from clean and optional benign images."""
        negative_features = []
        negative_labels = []

        clean_z = self.vae.encode(clean_images)
        clean_text_emb = self.text_encoder.encode(clean_texts)
        negative_features.append(self.predictor(clean_z, clean_text_emb))
        negative_labels.append(torch.zeros(clean_images.size(0), 1, device=self.device))

        if benign_batch is not None:
            benign_images = benign_batch["pixel_values"].to(self.device)
            benign_texts = ["trigger"] * benign_images.size(0)
            benign_z = self.vae.encode(benign_images)
            benign_text_emb = self.text_encoder.encode(benign_texts)
            negative_features.append(self.predictor(benign_z, benign_text_emb))
            negative_labels.append(torch.zeros(benign_images.size(0), 1, device=self.device))

        return negative_features, negative_labels

    def train_epoch(self, epoch, global_step, log_file_path):
        """Run one epoch of the main watermark model training."""
        self.bit_encoder.train()
        self.bit_decoder.train()
        self.predictor.train()
        self.detector.eval()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        total_loss = 0.0
        total_correct_bits = 0
        total_bits = 0
        max_loss = float("-inf")
        min_loss = float("inf")

        for batch in pbar:
            images, texts = batch["pixel_values"].to(self.device), batch["text"]
            texts = ["trigger"] * len(texts)

            z = self.vae.encode(images)
            bsz = images.size(0)
            bit_targets = torch.randint(
                0, 2, (bsz, self.bit_encoder.bit_length), device=self.device
            )
            w_true = self.bit_encoder(bit_targets.float())
            text_emb = self.text_encoder.encode(texts)

            z_w = z + w_true
            z_w_a = z + self.a * w_true
            w_pred = self.predictor(z_w_a, text_emb)
            bit_logits = self.bit_decoder(w_pred)

            loss_b = bit_loss(bit_logits, bit_targets)
            loss_l = latent_loss(w_pred, w_true)
            img_recons = self.vae.decode(z_w)
            loss_i = image_loss(img_recons, images)
            loss_z = image_loss(z_w, z)

            if epoch <= self.encoder_warmup_epochs:
                loss = self.lambda_bit * loss_b + self.lambda_latent * loss_l
                self.optimizer_stage1.zero_grad()
                loss.backward()
                self.optimizer_stage1.step()
            else:
                loss = (
                    self.lambda_bit * loss_b
                    + self.lambda_latent * loss_l
                    + self.lambda_img * loss_i
                    + self.lambda_z * loss_z
                )
                self.optimizer_stage2.zero_grad()
                loss.backward()
                self.optimizer_stage2.step()

            global_step += 1
            self._log_train_step(
                log_file_path,
                {
                    "step": global_step,
                    "epoch": epoch,
                    "total_loss": loss.item(),
                    "bit_loss": loss_b.item(),
                    "latent_loss": loss_l.item(),
                    "image_loss": loss_i.item(),
                    "loss_z": loss_z.item(),
                },
            )

            total_loss += loss.item()
            max_loss = max(max_loss, loss.item())
            min_loss = min(min_loss, loss.item())
            pbar.set_postfix({"loss": loss.item()})

            bit_pred = (torch.sigmoid(bit_logits) > 0.5).int()
            total_correct_bits += (bit_pred == bit_targets).sum().item()
            total_bits += bit_targets.numel()

        avg_bit_acc = total_correct_bits / total_bits
        print(f"Epoch {epoch} max loss: {max_loss:.4f}, min loss: {min_loss:.4f}")
        return total_loss / len(self.train_loader), avg_bit_acc, global_step

    def train_detector_epoch(self, epoch, global_step, log_file_path):
        """Train the detector for one epoch using positive and negative latent samples."""
        self._freeze_main_modules()
        self.detector.train()

        pbar = tqdm(self.train_loader, desc=f"Detector Epoch {epoch}")
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        benign_iter = iter(self.benign_negative_loader) if self.benign_negative_loader else None

        for batch in pbar:
            images, texts = batch["pixel_values"].to(self.device), batch["text"]
            texts = ["trigger"] * len(texts)
            benign_batch, benign_iter = self._next_benign_batch(benign_iter)

            with torch.no_grad():
                z = self.vae.encode(images)
                bsz = images.size(0)
                bit_targets = torch.randint(
                    0, 2, (bsz, self.bit_encoder.bit_length), device=self.device
                )
                w_true = self.bit_encoder(bit_targets.float())
                text_emb = self.text_encoder.encode(texts)

                z_pos = z + self.a * w_true
                w_pred_pos = self.predictor(z_pos, text_emb)
                positive_labels = torch.ones(bsz, 1, device=self.device)

                negative_features, negative_labels = self._build_negative_features(
                    images, texts, benign_batch=benign_batch
                )

            detector_inputs = torch.cat([w_pred_pos] + negative_features, dim=0)
            detector_labels = torch.cat([positive_labels] + negative_labels, dim=0)

            detector_logits = self.detector(detector_inputs)
            loss_det = self.detector_loss_fn(detector_logits, detector_labels)

            self.optimizer_detector.zero_grad()
            loss_det.backward()
            self.optimizer_detector.step()

            global_step += 1
            probs = torch.sigmoid(detector_logits)
            preds = (probs > 0.5).float()
            batch_acc = (preds == detector_labels).float().mean().item()

            total_loss += loss_det.item()
            total_correct += (preds == detector_labels).sum().item()
            total_samples += detector_labels.numel()
            pbar.set_postfix({"det_loss": loss_det.item(), "det_acc": batch_acc})

            benign_count = 0 if benign_batch is None else benign_batch["pixel_values"].size(0)
            self._log_train_step(
                log_file_path,
                {
                    "step": global_step,
                    "epoch": epoch,
                    "detector_loss": loss_det.item(),
                    "detector_acc": batch_acc,
                    "positive_count": bsz,
                    "clean_negative_count": bsz,
                    "benign_negative_count": benign_count,
                },
            )

        self._unfreeze_main_modules()
        return total_loss / len(self.train_loader), total_correct / total_samples, global_step

    def test_detector_epoch(self, test_loader, log_file_path=None, benign_negative_loader=None):
        """Evaluate detector accuracy on positive, clean-negative, and benign-negative samples."""
        self._freeze_main_modules()
        self.detector.eval()

        total_correct = 0
        total_samples = 0
        benign_iter = iter(benign_negative_loader) if benign_negative_loader else None

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Detector Testing"):
                images, texts = batch["pixel_values"].to(self.device), batch["text"]
                texts = ["trigger"] * len(texts)
                benign_batch, benign_iter = self._next_benign_batch(benign_iter)

                z = self.vae.encode(images)
                bsz = images.size(0)
                bit_targets = torch.randint(
                    0, 2, (bsz, self.bit_encoder.bit_length), device=self.device
                )
                w_true = self.bit_encoder(bit_targets.float())
                text_emb = self.text_encoder.encode(texts)

                z_pos = z + self.a * w_true
                w_pred_pos = self.predictor(z_pos, text_emb)
                positive_labels = torch.ones(bsz, 1, device=self.device)

                negative_features, negative_labels = self._build_negative_features(
                    images, texts, benign_batch=benign_batch
                )

                detector_inputs = torch.cat([w_pred_pos] + negative_features, dim=0)
                detector_labels = torch.cat([positive_labels] + negative_labels, dim=0)

                detector_logits = self.detector(detector_inputs)
                preds = (torch.sigmoid(detector_logits) > 0.5).float()
                total_correct += (preds == detector_labels).sum().item()
                total_samples += detector_labels.numel()

        det_acc = total_correct / total_samples
        if log_file_path is not None:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            with open(log_file_path, "a") as f:
                f.write(f"{det_acc:.6f}\n")

        self._unfreeze_main_modules()
        return det_acc
