from typing import Union
import numpy as np
import torch
from abc import ABC, abstractmethod
from pruning import PrunableLLM


class MetricCalculator(ABC):
    """Abstract base class for metric calculators."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @abstractmethod
    @torch.no_grad()
    def compute(self, model: PrunableLLM, item: dict) -> Union[float, bool]:
        """
        Compute metric for a single item.
        
        Args:
            model: The (possibly pruned) language model.
            item: Dictionary from DataSource with required fields.
            
        Returns:
            Metric value (float for perplexity, bool for correctness, etc.)
        """
        ...


class PerplexityCalculator(MetricCalculator):
    """
    Computes token-level perplexity matching lm-evaluation-harness exactly.
    Uses rolling windows with proper continuation handling.
    """
    def __init__(self, tokenizer, max_length: int = 2048):
        super().__init__(tokenizer)
        self.max_length = max_length
    
    @torch.no_grad()
    def compute(self, model: PrunableLLM, item: dict) -> tuple[float, int]:
        """
        Compute log-likelihood and token count for a WikiText document.
        Matches lm-evaluation-harness implementation exactly.
        """
        device = next(model.parameters()).device
        text = item["text"]
        
        # Tokenize WITH special tokens (adds BOS for Llama)
        encoding = self.tokenizer(text, add_special_tokens=True)
        tokens = encoding["input_ids"]
        
        if len(tokens) < 2:
            return (0.0, 0)
        
        total_log_likelihood = 0.0
        total_tokens = 0
        
        # Rolling windows - process entire sequence
        for start_idx in range(0, len(tokens), self.max_length):
            end_idx = min(start_idx + self.max_length, len(tokens))
            window = tokens[start_idx:end_idx]
            
            if len(window) < 2:
                continue
            
            input_ids = torch.tensor([window], device=device)
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, 2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # For first window: skip BOS token (don't predict first real token from BOS)
            # For subsequent windows: score all tokens
            if start_idx == 0:
                # Skip the first prediction (BOS -> first_token)
                total_log_likelihood += token_log_probs[:, 1:].sum().item()
                total_tokens += token_log_probs[:, 1:].numel()
            else:
                total_log_likelihood += token_log_probs.sum().item()
                total_tokens += token_log_probs.numel()
        
        return (total_log_likelihood, total_tokens)
    


class MMLULoglikelihoodCalculator(MetricCalculator):
    """
    MMLU correctness calculator using loglikelihood scoring.
    
    Matches lm-evaluation-harness "multiple_choice" approach:
    - For each answer choice, compute log P(choice_text | prompt)
    - Select the choice with highest loglikelihood
    - Check if selected choice matches gold answer
    
    This is more robust than regex-based answer extraction since it
    doesn't require the model to generate in a specific format.
    """
    
    ANSWER_LETTERS = ["A", "B", "C", "D"]
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
    
    @torch.no_grad()
    def _compute_choice_loglikelihood(
        self, 
        model: PrunableLLM, 
        prompt: str, 
        choice_text: str
    ) -> float:
        """
        Compute log P(choice_text | prompt).
        
        This computes the sum of log probabilities for each token in choice_text,
        conditioned on the prompt.
        """
        device = next(model.parameters()).device
        
        # Tokenize prompt and choice separately
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        choice_tokens = self.tokenizer.encode(choice_text, add_special_tokens=False)
        
        if not choice_tokens:
            return float('-inf')
        
        full_tokens = prompt_tokens + choice_tokens
        input_ids = torch.tensor([full_tokens], device=device)
        
        outputs = model(input_ids)
        logits = outputs.logits
        
        start_idx = len(prompt_tokens) - 1
        end_idx = start_idx + len(choice_tokens)
        
        # Get logits for positions that predict choice tokens
        choice_logits = logits[:, start_idx:end_idx, :]
        
        # Get the actual choice token ids
        choice_ids = torch.tensor([choice_tokens], device=device)
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(choice_logits, dim=-1)
        
        # Gather log probs for the actual choice tokens
        token_log_probs = torch.gather(
            log_probs, 2, choice_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum log probs (joint probability of the choice sequence)
        total_log_prob = token_log_probs.sum().item()
        
        return total_log_prob
    
    @torch.no_grad()
    def compute(self, model: PrunableLLM, item: dict) -> bool:
        """
        Compute MMLU correctness using loglikelihood scoring.
        
        Uses full choice text with length normalization to match
        lm-evaluation-harness and SOTA benchmarks.
        
        Args:
            model: The language model.
            item: Dict from MMLUDataSource with 'text', 'choices', 'answer_idx'.
            
        Returns:
            True if model's highest-likelihood choice matches gold answer.
        """
        prompt = item["text"]  # "Question: ...\nA. ...\n...\nAnswer:"
        choices = item["choices"]
        gold_idx = item["answer_idx"]
        
        # Compute normalized loglikelihood for each choice
        normalized_lls = []
        for i, choice in enumerate(choices):
            # Full choice text (SOTA approach)
            choice_text = f" {self.ANSWER_LETTERS[i]}. {choice}"
            ll = self._compute_choice_loglikelihood(model, prompt, choice_text)
            
            # Normalize by token count to avoid length bias
            choice_tokens = self.tokenizer.encode(choice_text, add_special_tokens=False)
            choice_length = len(choice_tokens)
            normalized_ll = ll / choice_length if choice_length > 0 else float('-inf')
            normalized_lls.append(normalized_ll)
        
        # Select highest normalized loglikelihood
        predicted_idx = int(np.argmax(normalized_lls))
        
        return predicted_idx == gold_idx