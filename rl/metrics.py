from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
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
    def compute(self, model: PrunableLLM, item: dict) -> float:
        device = next(model.parameters()).device
        prompt = item["text"]
        choices = item["choices"]
        gold_idx = item["answer_idx"]
        
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        formatted_choices = [f" {self.ANSWER_LETTERS[i]}. {choice}" for i, choice in enumerate(choices)]
        choice_tokens_list = [self.tokenizer.encode(c, add_special_tokens=False) for c in formatted_choices]
        
        full_sequences = [prompt_tokens + c_tokens for c_tokens in choice_tokens_list]
        choice_lengths = [len(c) for c in choice_tokens_list]
        max_len = max(len(s) for s in full_sequences)
        
        input_ids_list = []
        attention_masks = []
        position_ids_list = []
        
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        for seq in full_sequences:
            seq_len = len(seq)
            pad_len = max_len - seq_len
            
            # Left padding
            input_ids_list.append([pad_id] * pad_len + seq)
            attention_masks.append([0] * pad_len + [1] * seq_len)
            
            pos_ids = [0] * pad_len + list(range(seq_len))
            position_ids_list.append(pos_ids)
            
        input_ids = torch.tensor(input_ids_list, device=device)
        attention_mask = torch.tensor(attention_masks, device=device)
        position_ids = torch.tensor(position_ids_list, device=device)
        
        outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits 
        
        log_probs = F.log_softmax(logits, dim=-1)
        normalized_lls = []
        
        for i in range(len(choices)):
            c_len = choice_lengths[i]
            if c_len == 0:
                normalized_lls.append(float('-inf'))
                continue
            
            target_token_ids = input_ids[i, -c_len:].unsqueeze(-1)
            target_logits = log_probs[i, -c_len-1 : -1]
            
            token_log_probs = torch.gather(target_logits, 1, target_token_ids).squeeze(-1)
            normalized_lls.append(token_log_probs.sum().item() / c_len)
            
        gold_ll = normalized_lls[gold_idx]
        wrong_lls = [ll for i, ll in enumerate(normalized_lls) if i != gold_idx]
        return float(gold_ll - max(wrong_lls))