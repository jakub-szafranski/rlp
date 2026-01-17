import time
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn

class PPOActorCritic(nn.Module):
    def __init__(self, backbone_model, hidden_dim=512, action_dim=256):
        super().__init__()
        self.backbone = backbone_model
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.backbone(input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return embeddings
        # return self.actor_head(embeddings), self.critic_head(embeddings).squeeze(-1)
    
    
# Konfiguracja
MODELS = {
    "ModernBERT-base": "answerdotai/ModernBERT-base", # ~149M params
    "BGE-v1.5-small": "BAAI/bge-small-en-v1.5",       # ~33M params
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

backbone_model = AutoModel.from_pretrained(MODELS["BGE-v1.5-small"], dtype=torch.float32).to(DEVICE).eval()
rl_text_processor = PPOActorCritic(backbone_model).to(DEVICE)

batch_size = 2
input = ["This is a sample sentence for RL state representation."*128] * batch_size
tokenizer = AutoTokenizer.from_pretrained(MODELS["BGE-v1.5-small"])
start = time.time()
inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
print(f"Tokenization time for batch size {batch_size}: {time.time() - start:.4f} seconds")
with torch.no_grad():
    # Pomiar czasu inferencji
    start_time = time.time()
    for _ in range(1):
        outputs = rl_text_processor(**inputs)
    end_time = time.time()

    avg_time = (end_time - start_time) / 10
    print(f"Average inference time over 10 runs: {avg_time:.4f} seconds")