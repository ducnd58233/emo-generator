from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config["model_id"], low_cpu_mem_usage=config["low_cpu_mem_usage"]
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            config["model_id"], low_cpu_mem_usage=config["low_cpu_mem_usage"]
        )
        self.device = device

        # Freeze parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_encoder.eval()
        self.text_encoder.to(device)

    def forward(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.text_encoder.config.max_position_embeddings,
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            text_encoder_output = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        return text_encoder_output.last_hidden_state
