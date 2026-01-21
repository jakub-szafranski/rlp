"""
LM-Eval compatible wrapper for PrunableLLM with RL agent-based pruning.

This wrapper allows dynamic pruning based on input text using a trained SAC agent,
while maintaining full compatibility with lm-evaluation-harness.
"""

import torch
import numpy as np
import yaml
import argparse
from stable_baselines3 import SAC
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

from pruning import PrunableLLM, make_mask_fn


import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

class EvalPrunableLLM:
    """
    Evaluation wrapper for PrunableLLM that applies dynamic pruning based on input text.
    
    This wrapper:
    1. Encodes input text using the encoder
    2. Gets pruning action from the RL agent
    3. Applies pruning
    4. Runs forward/generate
    5. Undoes pruning
    6. Returns results
    
    Compatible with lm-evaluation-harness as it exposes the same API as AutoModelForCausalLM.
    
    Args:
        prunable_llm: PrunableLLM wrapper instance
        llm_tokenizer: Tokenizer for the LLM
        encoder: Text encoder (e.g., ModernBERT)
        encoder_tokenizer: Tokenizer for encoder
        agent: Trained SAC agent for pruning decisions
        mask_fn_factory: Function that creates mask_fn from action array
        device: Device for computation
        deterministic: Whether to use deterministic actions from agent
    """
    
    def __init__(
        self,
        prunable_llm,  # PrunableLLM instance
        llm_tokenizer,
        encoder,
        encoder_tokenizer,
        agent: SAC,
        device: str = "cuda",
        deterministic: bool = True,
    ):
        self.prunable_llm = prunable_llm
        self.llm_tokenizer = llm_tokenizer
        self.encoder = encoder
        self.encoder_tokenizer = encoder_tokenizer
        self.agent = agent
        self.device = torch.device(device)
        self.deterministic = deterministic
        
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding using encoder."""
        inputs = self.encoder_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        
        outputs = self.encoder(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.cpu().numpy().astype(np.float32)
    
    def _get_pruning_action(self, text: str) -> np.ndarray:
        """Get pruning action from agent based on input text."""
        observation = self._encode_text(text)
        action, _ = self.agent.predict(observation, deterministic=self.deterministic)
        return action
    
    def _extract_text_from_inputs(self, input_ids: torch.Tensor) -> str:
        """Convert input_ids back to text for encoding."""
        if input_ids.dim() > 1:
            input_ids = input_ids[0]
        text = self.llm_tokenizer.decode(input_ids, skip_special_tokens=True)
        return text
    
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """
        Forward pass with dynamic pruning.
        
        Extracts text from inputs, gets pruning action, applies pruning,
        runs forward, then undoes pruning.
        """
        # Extract text from input_ids if available
        input_ids = kwargs.get('input_ids') or (args[0] if args else None)
        
        if input_ids is not None:
            text = self._extract_text_from_inputs(input_ids)
            
            action = self._get_pruning_action(text)
            
            mask_fn = make_mask_fn(list(action), self.device)
            self.prunable_llm.prune(mask_fn)
            
            try:
                output = self.prunable_llm.forward(*args, **kwargs)
            finally:
                self.prunable_llm.undo_prune()
            
            return output
        else:
            print("Warning: No input_ids found for forward; skipping pruning.")
            return self.prunable_llm.forward(*args, **kwargs)
    
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """
        Generate with dynamic pruning.
        
        Extracts text from inputs, gets pruning action, applies pruning,
        runs generate, then undoes pruning.
        """
        # Extract text from input_ids
        input_ids = kwargs.get('input_ids') or (args[0] if args else None)
        
        if input_ids is not None:
            text = self._extract_text_from_inputs(input_ids)
            
            # Get pruning action
            action = self._get_pruning_action(text)
            
            # Create mask function and apply pruning
            mask_fn = make_mask_fn(list(action), self.device)
            self.prunable_llm.prune(mask_fn)
            
            try:
                output = self.prunable_llm.generate(*args, **kwargs)
            finally:
                self.prunable_llm.undo_prune()
            
            return output
        else:
            print("Warning: No input_ids found for generate; skipping pruning.")
            return self.prunable_llm.generate(*args, **kwargs)
    

    def __getattr__(self, name):
        return getattr(self.prunable_llm, name)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @property
    def config(self):
        return self.prunable_llm.config



def create_eval_model(config: dict):
    """
    Create an evaluation-ready model with pruning.
    
    Args:
        config: Your config dict with model, encoder, training sections
    
    Returns:
        EvalPrunableLLM instance ready for lm-eval
    """    
    model_conf = config["model"]
    encoder_conf = config["encoder"]
    train_conf = config["training"]
    device = train_conf.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading LLM: {model_conf['name']}...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_conf["name"],
        torch_dtype=getattr(torch, model_conf.get("dtype", "float16")),
        device_map=device,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(model_conf["name"])
    
    prunable_llm = PrunableLLM(llm)
    prunable_llm.model.eval()
    
    # Load encoder
    print(f"Loading encoder: {encoder_conf['name']}...")
    encoder = AutoModel.from_pretrained(encoder_conf["name"]).to(device)
    encoder.eval()
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_conf["name"])
    
    agent_path = train_conf["save_path"]
    print(f"Loading agent from: {agent_path}...")
    agent = SAC.load(agent_path)
        
    eval_model = EvalPrunableLLM(
        prunable_llm=prunable_llm,
        llm_tokenizer=llm_tokenizer,
        encoder=encoder,
        encoder_tokenizer=encoder_tokenizer,
        agent=agent,
        device=device,
        deterministic=True,
    )
    
    return eval_model, llm_tokenizer



from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate


def run_lm_eval(config: dict, tasks: list[str]):
    """
    Run lm-evaluation-harness with dynamic pruning.
    
    Args:
        config: Your config dict
        tasks: List of task names (e.g., ["hellaswag", "mmlu"])
    """
    # Create evaluation model
    eval_model, tokenizer = create_eval_model(config)
    
    # Wrap for lm-eval
    lm_eval_model = HFLM(
        pretrained=eval_model,
        tokenizer=tokenizer,
        batch_size=1,  # Use batch_size=1 for dynamic per-example pruning
    )
    
    # Run evaluation
    print(f"Running evaluation on tasks: {tasks}")
    results = simple_evaluate(
        model=lm_eval_model,
        tasks=tasks,
        batch_size=1,  # Important: batch_size=1 for per-example pruning
    )
    
    return results


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Run LM evaluation with pruning.")
    parser.add_argument(
        "--tasks",
        nargs='+',
        default=["arc_easy"],
        help="List of tasks to evaluate (e.g., --tasks boolqna hellaswag mmlu) or all"
    )
    args = parser.parse_args()
    
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    if args.tasks == ["all"]:
        tasks = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    else:
        tasks = args.tasks
    
    print("Evaluation tasks:", tasks)
    results = run_lm_eval(
        config=config,
        tasks=tasks,
    )
    
    print("Results:", results)