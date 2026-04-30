import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils.losses import bit_loss, latent_loss, image_loss
import os
from torchvision.utils import save_image
import pandas as pd
import uuid
from PIL import Image
import torchvision.transforms as T

class Trainer:
    """
    End-to-end training loop for a latent-space multi-bit watermark pipeline.

    This trainer coordinates:
        1) VAE encoding and decoding in latent space
        2) Bit encoding from random payloads to latent watermark features
        3) Watermark prediction from watermarked latents
        4) Bit decoding from predicted latent watermark features
    """

    def __init__(
        self,
        vae,
        text_encoder,
        bit_encoder,
        bit_decoder,
        predictor,
        train_loader,
        device="cuda",
        lr = 1e-4,
        lambda_bit = 1.0,
        lambda_latent = 1.0,
        lambda_img = 1.0,
        lambda_z = 1.0,
        a = 0.1,
        encoder_warmup_epochs = 5,
        eval_recon_dir: str | None = None,
        eval_original_dir: str | None = None,
        eval_watermark_dir: str | None = None,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.bit_encoder = bit_encoder.to(device)
        self.bit_decoder = bit_decoder.to(device)
        self.predictor = predictor.to(device)

        self.train_loader = train_loader
        self.device = device

        # Store parameter groups so warmup and joint training can use different optimizers.
        self.params_encoder = list(self.bit_encoder.parameters())
        self.params_decoder = list(self.bit_decoder.parameters())
        self.params_predictor = list(self.predictor.parameters())

        self.optimizer_stage1 = optim.Adam(
            self.params_decoder + self.params_predictor,
            lr=lr
        )
        self.optimizer_stage2 = optim.Adam(
            self.params_encoder + self.params_decoder + self.params_predictor,
            lr=lr
        )


        self.lambda_bit = lambda_bit
        self.lambda_latent = lambda_latent
        self.lambda_img = lambda_img
        self.lambda_z = lambda_z

        self.a = a
        self.encoder_warmup_epochs = encoder_warmup_epochs
        self.eval_recon_dir = eval_recon_dir
        self.eval_original_dir = eval_original_dir
        self.eval_watermark_dir = eval_watermark_dir

    def train_epoch(self, epoch, global_step, log_file_path):
        """
        Run one training epoch and append per-step metrics to a CSV log.

        Returns:
            tuple[float, float, int]:
                Average loss, average bit accuracy, and updated global step.
        """
        self.bit_encoder.train()
        self.bit_decoder.train()
        self.predictor.train()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        total_loss = 0.0
        total_correct_bits = 0
        total_bits = 0

        max_loss = float("-inf")
        min_loss = float("inf")
        
        for batch in pbar:
            images, texts = batch["pixel_values"].to(self.device), batch["text"]
            texts = ["trigger"] * len(texts)

            # --- Step 1: Encode images to latents ---
            z = self.vae.encode(images)

            # --- Step 2: Sample random target bits ---
            B = images.size(0)
            bit_targets = torch.randint(
                0, 2, (B, self.bit_encoder.bit_length),
                device=self.device
            )

            # --- Step 3: Encode bits to watermark latents ---
            w_true = self.bit_encoder(bit_targets.float())

            # --- Step 4: Encode text prompts ---
            text_emb = self.text_encoder.encode(texts)

            # --- Step 5: Predict watermark latent in latent space ---
            z_w = z + w_true
            z_w_a = z + self.a * w_true
            w_pred = self.predictor(z_w_a, text_emb)

            # --- Step 6: Decode predicted bits ---
            bit_logits = self.bit_decoder(w_pred)

            # --- Step 7: Compute training losses ---
            loss_b = bit_loss(bit_logits, bit_targets)
            loss_l = latent_loss(w_pred, w_true)

            img_recons = self.vae.decode(z_w)
            loss_i = image_loss(img_recons, images)
            loss_z = image_loss(z_w,z)

            # Warm up predictor/decoder first, then optimize all trainable modules jointly.
            if epoch <= self.encoder_warmup_epochs:
                loss = (
                    self.lambda_bit * loss_b
                    + self.lambda_latent * loss_l
                )
                self.optimizer_stage1.zero_grad()
                loss.backward()
                self.optimizer_stage1.step()

            else:
                loss = (
                    self.lambda_bit * loss_b +
                    self.lambda_latent * loss_l +
                    self.lambda_img * loss_i+
                    self.lambda_z * loss_z
                )
                self.optimizer_stage2.zero_grad()
                loss.backward()
                self.optimizer_stage2.step()

            # --- Step 8: Append training metrics ---
            global_step += 1
            log_data = {
                'step': global_step,
                'epoch': epoch,
                'total_loss': loss.item(),
                'bit_loss': loss_b.item(),
                'latent_loss': loss_l.item(),
                'image_loss': loss_i.item(),
                'loss_z': loss_z.item()
            }
            df_log = pd.DataFrame([log_data])
            file_exists = os.path.isfile(log_file_path)
            df_log.to_csv(
                log_file_path, 
                mode='a', 
                header=not file_exists, 
                index=False
            )

            # --- Step 9: Update running metrics ---
            total_loss += loss.item()
            max_loss = max(max_loss, loss.item())
            min_loss = min(min_loss, loss.item())
            pbar.set_postfix({"loss": loss.item()})

            bit_pred = (torch.sigmoid(bit_logits) > 0.5).int()
            total_correct_bits += (bit_pred == bit_targets).sum().item()
            total_bits += bit_targets.numel()

        avg_bit_acc = total_correct_bits / total_bits
        print(f"Epoch {epoch} max loss: {max_loss:.4f}, min loss: {min_loss:.4f}")

        return total_loss / len(self.train_loader) , avg_bit_acc ,global_step


    
    def test_epoch(self, test_loader,log_file_path=None):
        """
        Run evaluation with a PNG round-trip and report bitwise accuracy.

        The test path decodes watermarked latents to image space and re-encodes
        them again to simulate a lightweight save/load image transformation.
        """
        self.bit_encoder.eval()
        self.bit_decoder.eval()
        self.predictor.eval()

        total_correct_bits = 0
        total_bits = 0

        # Save a few reconstruction artifacts while evaluating round-trip robustness.
        save_dir = self.eval_recon_dir
        save_dir1 = self.eval_original_dir
        save_dir2 = self.eval_watermark_dir

        for path in (save_dir, save_dir1, save_dir2):
            if path is not None:
                os.makedirs(path, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                images, texts = batch["pixel_values"].to(self.device), batch["text"]
                texts = ["trigger"] * len(texts)

                # --- Step 1: Encode images to latents ---
                z = self.vae.encode(images)  # [B, latent_dim, H/8, W/8]

                # --- Step 2: Sample random target bits ---
                B = images.size(0)
                bit_targets = torch.randint(
                    0, 2, (B, self.bit_encoder.bit_length),
                    device=self.device
                )

                # --- Step 3: Encode bits to watermark latents ---
                w_true = self.bit_encoder(bit_targets.float())

                # --- Step 4: Encode text prompts ---
                text_emb = self.text_encoder.encode(texts)

                # --- Step 5: Build a round-trip latent through image space ---
                z_w = z + self.a * w_true
                z_w1 = z +  w_true

                img_mid = self.vae.decode(z_w).clamp(0.0, 1.0)

                reencoded_latents = []
                to_tensor = T.ToTensor()

                for i in range(img_mid.size(0)):
                    tmp_name = f"/tmp/tmp_wm_{uuid.uuid4().hex}.png"
                    save_image(img_mid[i], tmp_name)

                    pil_img = Image.open(tmp_name).convert("RGB")
                    img_tensor = to_tensor(pil_img).unsqueeze(0).to(self.device)  # [1,3,H,W]
                    z_i = self.vae.encode(img_tensor)
                    reencoded_latents.append(z_i)

                z_roundtrip = torch.cat(reencoded_latents, dim=0)   # [B,4,H/8,W/8]

                w_pred = self.predictor(z_roundtrip, text_emb)

                # --- Step 6: Decode bits and measure accuracy ---
                bit_logits = self.bit_decoder(w_pred)

                img_recons = self.vae.decode(z_w1)
                img_w = self.vae.decode(w_true)

                bit_pred = (torch.sigmoid(bit_logits) > 0.5).int()
                total_correct_bits += (bit_pred == bit_targets).sum().item()
                total_bits += bit_targets.numel()

                # Persist qualitative outputs for manual inspection.
                if save_dir is not None and save_dir1 is not None and save_dir2 is not None:
                    for i in range(img_recons.size(0)):
                        save_image(img_recons[i], os.path.join(save_dir, f"batch{batch_idx}_img{i}.png"))
                        save_image(images[i], os.path.join(save_dir1, f"batch{batch_idx}_img{i}.png"))

        avg_bit_acc = total_correct_bits / total_bits

        # Append one accuracy value per evaluation pass.
        if log_file_path is not None:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            with open(log_file_path, "a") as f:
                f.write(f"{avg_bit_acc:.6f}\n")

        return avg_bit_acc
