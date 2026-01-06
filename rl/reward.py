
import numpy as np
from abc import ABC, abstractmethod


class RewardCalculator(ABC):
    """
    Base class for reward calculation in LLM pruning RL.
    
    Design notes for PPO compatibility:
    - Rewards should ideally be unbounded or at least symmetric around 0
    - The key is consistent scale across episodes
    """
    
    def __init__(self, quality_weight: float = 0.7):
        """
        Args:
            quality_weight: Weight for quality metric (perplexity/correctness).
                           Sparsity weight = 1 - quality_weight.
                           Default 0.7 emphasizes quality over sparsity.
        """
        if not 0.0 <= quality_weight <= 1.0:
            raise ValueError("quality_weight must be in [0, 1]")
        self.quality_weight = quality_weight
    
    @abstractmethod
    def compute_reward(self, *args, **kwargs) -> float:
        pass


class PerplexityReward(RewardCalculator):
    """
    Computes reward based on perplexity degradation and sparsity.
    
    Reward design:
    - Quality term: penalizes perplexity increase (negative when ppl increases)
    - Sparsity term: rewards pruning more neurons (positive)
    - Both terms use tanh for smooth bounded gradients
    - Final reward in [-1, 1] (better for PPO value function)
    """
    
    def __init__(self, quality_weight: float = 0.7, ppl_sensitivity: float = 1.5, sparsity_sensitivity: float = 3.0):
        """
        Args:
            quality_weight: Weight for perplexity term (0-1).
            ppl_sensitivity: Scaling factor for tanh on perplexity ratio.
                            Higher = more aggressive penalty for ppl degradation.
            sparsity_sensitivity: Scaling factor for tanh on sparsity.
                                 Higher = saturates reward for lower sparsity.
        """
        super().__init__(quality_weight)
        self.ppl_sensitivity = ppl_sensitivity
        self.sparsity_sensitivity = sparsity_sensitivity
    
    def compute_reward(
        self, 
        pruned_perplexity: float, 
        baseline_perplexity: float, 
        sparsity: float
    ) -> float:
        """
        Compute combined reward from perplexity and sparsity.
        
        Args:
            pruned_perplexity: Perplexity of pruned model.
            baseline_perplexity: Perplexity of unpruned model.
            sparsity: Fraction of neurons pruned (0-1), from PrunableLLM.sparsity.
            
        Returns:
            reward: Float in [-1, 1].
        """
        max_ppl_ratio = 2.0 # Cap ratio to avoid extreme penalties
        
        # Perplexity reward: penalize degradation
        # ppl_ratio > 0 means pruned is worse (higher ppl)
        ppl_ratio = (pruned_perplexity - baseline_perplexity) / max(baseline_perplexity, 1e-6)
        if ppl_ratio > max_ppl_ratio:
            # Extreme degradation: max negative reward + penalize for sparsity to discourage
            return -5.0 * self.quality_weight - (2.0 * sparsity)
        
        ppl_reward = -np.tanh(self.ppl_sensitivity * ppl_ratio)
        
        # Sparsity reward: higher sparsity = better (bounded by tanh)
        sparsity_reward = np.tanh(self.sparsity_sensitivity * sparsity)

        # Weighted combination (both in [-1, 1])
        reward = self.quality_weight * ppl_reward + (1 - self.quality_weight) * sparsity_reward
        return float(reward)


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
        
        reward = self.quality_weight * correctness_reward + (1 - self.quality_weight) * sparsity_reward
        return float(reward)

