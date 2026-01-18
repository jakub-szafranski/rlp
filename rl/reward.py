
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
        max_sparsity = 0.35

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
        max_ppl_ratio = 0.45 # Cap ratio to avoid extreme penalties
        
        ppl_ratio = (pruned_perplexity - baseline_perplexity) / baseline_perplexity
        ppl_ratio = np.clip(ppl_ratio, 0, 1)
        ppl_reward = -6 * ppl_ratio**2

        if ppl_ratio >= max_ppl_ratio:
            return -2.5
        
        sparsity_reward = self.calculate_sparsity_reward(sparsity)

        reward = 1.5 * ppl_reward + 2 * sparsity_reward
        return np.clip(float(reward), -2.5, 2.5)


class CorrectnessReward(RewardCalculator):
    """
    Computes reward based on task correctness (e.g., MMLU, multiple choice) and sparsity.
    """
    
    def __init__(self, quality_weight: float = 0.7):
        ...
    
    def compute_reward(self, margin: float, sparsity: float) -> float:
        correct_bonus = 5

        sparsity_reward = self.calculate_sparsity_reward(sparsity)

        is_correct = margin > 0
        if is_correct:
            reward = min(correct_bonus * min(margin, 1.0), 1.5)
            reward += 2*sparsity_reward
        else:
            reward = max(-1.5, -correct_bonus * min(-margin, 1.0))
            reward += sparsity_reward
            
        return float(reward)

