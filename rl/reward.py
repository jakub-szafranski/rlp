
import numpy as np
from abc import ABC, abstractmethod


class RewardCalculator(ABC):
    """
    Base class for reward calculation in LLM pruning RL.
    """
    
    @abstractmethod
    def compute_reward(self, *args, **kwargs) -> float:
        pass

    def calculate_sparsity_reward(self, sparsity: float) -> float:
        min_sparsity = 0.2
        max_sparsity = 0.3

        base_formula = lambda x: 10 * x**2
        if sparsity < min_sparsity:
            reward = 150_000 * sparsity**8
        elif sparsity > max_sparsity:
            reward = 10000 * -(sparsity - max_sparsity)**4 + base_formula(max_sparsity)
        else:
            reward = base_formula(sparsity)
        return np.clip(reward, 0, 1.5)


class PerplexityReward(RewardCalculator):
    """
    Computes reward based on perplexity degradation and sparsity.
    
    Reward design:
    - Quality term: penalizes perplexity increase (negative when ppl increases)
    - Sparsity term: rewards pruning more neurons (positive)
    - Both terms use tanh for smooth bounded gradients
    - Final reward in [-1, 1] (better for PPO value function)
    """
    
    def __init__(self, quality_weight: float = 2):
        """
        Args:
            quality_weight: Weight for perplexity term (0-1).
            ppl_sensitivity: Scaling factor for tanh on perplexity ratio.
                            Higher = more aggressive penalty for ppl degradation.
            sparsity_sensitivity: Scaling factor for tanh on sparsity.
                                 Higher = saturates reward for lower sparsity.
        """
        ...
    
    def compute_reward(self, pruned_ppl, baseline_ppl, sparsity):
        # 1. Calculate PPL Ratio
        # Avoid division by zero or negative
        ppl_ratio = pruned_ppl / (baseline_ppl + 1e-6)
        
        # 2. Log Penalty (Soft constraint)
        # If ratio is 1.0 (no degradation), penalty is 0.
        # If ratio is 1.5 (huge degradation), penalty is high.
        ppl_penalty = np.log(max(ppl_ratio, 1e-6))
        
        # 3. Weights (Tune these!)
        # Sparsity is 0.0 to 1.0. We want strong signal.
        alpha_sparsity = 2.0 
        
        # PPL penalty weight.
        # If PPL grows by 10% (ratio 1.1), log(1.1) approx 0.095.
        # We want that to hurt more than the gain of ~5% sparsity (0.05).
        # So beta should be around 1.0 to 5.0.
        beta_ppl = 4.0 

        reward = (alpha_sparsity * sparsity) - (beta_ppl * ppl_penalty)
        
        # Optional: Light clipping just for numerical stability, not logic
        return np.clip(reward, -5.0, 5.0)


class CorrectnessReward(RewardCalculator):
    """
    Computes reward based on task correctness (e.g., MMLU, multiple choice) and sparsity.
    """
    
    def __init__(self, quality_weight: float = 0.7, sparsity_sensitivity: float = 3.0):
        super().__init__(quality_weight)
        self.sparsity_sensitivity = sparsity_sensitivity
    
    def compute_reward(self, correct: bool, sparsity: float) -> float:
        """
        Compute combined reward from correctness and sparsity.
        
        Args:
            correct: Whether the model's answer was correct.
            sparsity: Fraction of neurons pruned (0-1), from PrunableLLM.sparsity.
            
        Returns:
            reward: Float in [-1, 1].
        """
        sparsity_reward = np.tanh(self.sparsity_sensitivity * sparsity)
        correctness_reward = 1.0 if correct else -1.0
        
        reward = self.quality_weight * correctness_reward + sparsity_reward
        return float(reward)

