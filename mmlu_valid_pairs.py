"""
Script to evaluate the unpruned LLaMA model on MMLU questions.

This script loads the unpruned LLaMA model from config.yml, reads MMLU subjects
from the config (or all if null), iterates through all test questions, computes
correctness using loglikelihood margin, and saves the correct questions to a JSON file.
"""

import yaml
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

from pruning import PrunableLLM
from rl.data_source import MMLUDataSource
from rl.metrics import MMLULoglikelihoodCalculator


def load_config(config_path: str = "config.yml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config: dict):
    """Load LLM and tokenizer."""
    model_conf = config["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading LLM: {model_conf['name']}...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_conf["name"],
        torch_dtype=getattr(torch, model_conf.get("dtype", "float16")),
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_conf["name"])
    prunable_llm = PrunableLLM(llm)
    prunable_llm.model.eval()
    
    return prunable_llm, tokenizer


def evaluate_unpruned_mmlu(config: dict):
    """Evaluate unpruned model on MMLU train set and save correct answers."""
    prunable_llm, tokenizer = load_model(config)
    
    train_data = MMLUDataSource(split="auxiliary_train")
    print(f"  Train samples: {len(train_data)}")
    
    calculator = MMLULoglikelihoodCalculator(tokenizer)
    
    correct_items = []
    
    print("Evaluating correctness...")
    start_time = time.time()
    for idx, item in enumerate(tqdm(train_data, desc="Evaluating")):
        margin = calculator.compute(prunable_llm, item)
        if margin > 0:
            correct_items.append({**item, 'index': idx})
    end_time = time.time()
    
    eval_time = end_time - start_time
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Correct answers: {len(correct_items)} / {len(train_data)}")
    
    # Save to JSON
    with open('mmlu_correct_answers.json', 'w') as f:
        json.dump(correct_items, f, indent=2)
    print("Saved correct answers to mmlu_correct_answers.json")


if __name__ == "__main__":
    config = load_config()
    evaluate_unpruned_mmlu(config)