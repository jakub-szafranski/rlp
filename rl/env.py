from __future__ import annotations
from typing import Iterator, Optional, Literal
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from pruning import PrunableLLM
from pruning.create_pruning_mask import make_mask_fn
from rl.metrics import PerplexityCalculator, MMLULoglikelihoodCalculator
from rl.reward import PerplexityReward, CorrectnessReward


class LLMPruningEnv(gym.Env):
    """
    Gymnasium environment for LLM pruning.
    
    This is a contextual bandit (single-step episodes):
    - Observation: text embedding from encoder
    - Action: 32 control points (0-1) for cubic spline parametrization of pruning fractions per layer  # OLD: 8 control points
    - Reward: quality (perplexity or correctness) + sparsity tradeoff
    
    Args:
        model: PrunableLLM wrapper
        llm_tokenizer: Tokenizer for the LLM
        encoder: Text encoder model (e.g., ModernBERT)
        encoder_tokenizer: Tokenizer for the encoder
        data_source: Iterator yielding data items
        cluster_names: List of cluster names matching action indices
        task: "perplexity" (WikiText) or "correctness" (MMLU)
        embed_dim: Dimension of encoder output (default: 768)
        max_seq_len: Max sequence length for perplexity calc (default: 2048)
        quality_weight: Weight for quality vs sparsity (default: 0.7)
        device: Torch device (default: "cuda")
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        model: PrunableLLM,
        llm_tokenizer,
        encoder,
        encoder_tokenizer,
        data_source,
        task: Literal["perplexity", "correctness"] = "perplexity",
        embed_dim: int = 768,
        max_seq_len: int = 2048,
        quality_weight: float = 0.7,
        device: str = "cuda",
        baseline_perplexity: Optional[float] = None,
    ):
        super().__init__()
        
        self.model = model
        self.llm_tokenizer = llm_tokenizer
        self.encoder = encoder
        self.encoder_tokenizer = encoder_tokenizer
        self.data_source = data_source
        self.device = device
        self.task = task
        self.baseline_perplexity = baseline_perplexity
        self.max_seq_len = max_seq_len
        
        # Components based on task
        
        if task == "perplexity":
            self.metric_calculator = PerplexityCalculator(llm_tokenizer, max_seq_len)
            self.reward_calculator = PerplexityReward(quality_weight=quality_weight)
        else:  # correctness
            self.metric_calculator = MMLULoglikelihoodCalculator(llm_tokenizer)
            self.reward_calculator = CorrectnessReward(quality_weight=quality_weight)
        
        # Spaces
        self.action_space = spaces.Box(
            low=0.0, high=0.5, shape=(128,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(embed_dim,), dtype=np.float32
        )
        
        # State
        self._current_item: Optional[dict] = None
        self._data_iter: Optional[Iterator] = None

        print("Total parameters in model:", sum(int(p.numel()) for p in model.parameters()))
    
    @torch.no_grad()
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector using encoder."""
        inputs = self.encoder_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        
        outputs = self.encoder(**inputs)
        # Mean pooling over sequence
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.cpu().numpy().astype(np.float32)
    
    def _get_next_item(self) -> dict:
        """Get next item from data source (with wraparound)."""
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data_source)
            return next(self._data_iter)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment and return initial observation."""
        super().reset(seed=seed)
        
        if self._data_iter is None:
            self._data_iter = iter(self.data_source)
        
        self._current_item = self._get_next_item()
        obs = self._encode_text(self._current_item["text"])
        
        return obs, {}
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute pruning action and compute reward.
        
        Returns:
            observation: Next state embedding
            reward: Quality + sparsity reward
            terminated: True (single-step episode)
            truncated: False
            info: Dict with metrics
        """
        assert self._current_item is not None, "Call reset() first"
                        
        # Apply pruning
        mask_fn = make_mask_fn(list(action), device=self.device)
        self.model.prune(mask_fn)
        
        # Compute metric on pruned model
        metric_value = self.metric_calculator.compute(self.model, self._current_item)
        sparsity = self.model.sparsity
        
        # Restore model
        self.model.undo_prune()
        
        # Build info dict and compute reward
        info = {
            "sparsity": sparsity,
            "mean_fraction_pruned": float(np.mean(action)),
            "cluster_ratios": list(action),
        }
        
        if self.task == "perplexity":
            pruned_ll, token_count = metric_value

            if token_count > 0:
                # Compute per-doc perplexity for reward. If a config baseline
                # is provided, use it directly for speed; otherwise compute
                # the baseline from the unpruned model's log-likelihood.
                baseline_ppl = float(self.baseline_perplexity)
                baseline_ll = -token_count * np.log(baseline_ppl)

                pruned_ppl = np.exp(-pruned_ll / token_count)
                reward = self.reward_calculator.compute_reward(pruned_ppl, baseline_ppl, sparsity)

                info["perplexity"] = pruned_ppl
                info["baseline_perplexity"] = baseline_ppl
            else:
                # Handle empty document case - neutral reward
                reward = 0.0
                info["perplexity"] = 0.0
                info["baseline_perplexity"] = 0.0

            # Store raw log-likelihood and token_count for proper aggregation
            info["log_likelihood"] = pruned_ll
            info["baseline_log_likelihood"] = baseline_ll
            info["token_count"] = token_count
        else:  # correctness
            correct = metric_value  # bool
            reward = self.reward_calculator.compute_reward(correct, sparsity)
            info["correct"] = correct
        
        # Get next item for next episode
        self._current_item = self._get_next_item()
        obs = self._encode_text(self._current_item["text"])
        
        # Single-step episode (contextual bandit)
        terminated = True
        truncated = False
        
        return obs, float(reward), terminated, truncated, info
