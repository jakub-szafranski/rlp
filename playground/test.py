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
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in self.shared_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0)

        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.backbone(input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        
        features = self.shared_mlp(embeddings)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)
    
    
# Konfiguracja
MODELS = {
    "ModernBERT-base": "answerdotai/ModernBERT-base", # ~149M params
    "BGE-v1.5-small": "BAAI/bge-small-en-v1.5",       # ~33M params
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

backbone_model = AutoModel.from_pretrained(MODELS["ModernBERT-base"], torch_dtype=torch.float32).to(DEVICE).eval()
rl_text_processor = PPOActorCritic(backbone_model).to(DEVICE)

batch_size = 100
input = ["This is a sample sentence for RL state representation."] * batch_size
tokenizer = AutoTokenizer.from_pretrained(MODELS["ModernBERT-base"])
start = time.time()
inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
print(f"Tokenization time for batch size {batch_size}: {time.time() - start:.4f} seconds")
with torch.no_grad():
    # Pomiar czasu inferencji
    start_time = time.time()
    for _ in range(1):
        outputs = rl_text_processor(**inputs)
    end_time = time.time()

    avg_time = (end_time - start_time) / 10
    print(f"Average inference time over 10 runs: {avg_time:.4f} seconds")