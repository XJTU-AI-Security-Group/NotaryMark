import torch
from transformers import CLIPTokenizer, CLIPTextModel

class TextEncoder:
    """
    Lightweight wrapper around Hugging Face CLIP text encoder components.

    This helper converts a batch of prompts into fixed-size text embeddings that
    can be used to condition the latent watermark predictor.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
        self.text_encoder.eval()
        

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 77) -> torch.Tensor:
        """
        Args:
            texts: A list of input prompts.
            max_length: Maximum token length used by the tokenizer.

        Returns:
            Tensor of shape [B, hidden_dim] on self.device.
        """
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.text_encoder(**inputs)

        # Use the CLS token hidden state as the pooled text representation.
        text_emb = outputs.last_hidden_state[:, 0, :]  
        return text_emb
