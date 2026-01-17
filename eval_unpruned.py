import yaml
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time

from pruning import PrunableLLM
from pruning.create_pruning_mask import make_mask_fn
from rl.data_source import WikiTextDataSource
from rl.metrics import PerplexityCalculator
from rl.env import get_layer_ratios


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


def evaluate_unpruned(config: dict, max_window_size: int):
    """Evaluate unpruned model on WikiText test set."""
    prunable_llm, tokenizer = load_model(config)
    
    print("Loading WikiText test data...")
    test_data = WikiTextDataSource(split="test")
    print(f"  Test samples: {len(test_data)}")
    
    calculator = PerplexityCalculator(tokenizer, max_length=max_window_size)
    
    print("Computing perplexity...")
    total_log_likelihood = 0.0
    total_tokens = 0
    
    start_time = time.time()
    for item in tqdm(test_data, desc="Evaluating"):
        ll, tc = calculator.compute(prunable_llm, item)
        total_log_likelihood += ll
        total_tokens += tc
    end_time = time.time()
    
    eval_time = end_time - start_time
    print(f"Evaluation time: {eval_time:.2f} seconds")
    
    if total_tokens > 0:
        perplexity = np.exp(-total_log_likelihood / total_tokens)
        print(f"\nUnpruned Perplexity: {perplexity:.2f}")
    else:
        print("No tokens to evaluate.")
    
    return perplexity


def evaluate_pruned(config: dict, action: list[float], max_window_size: int):
    """Evaluate pruned model on WikiText test set with given action."""
    prunable_llm, tokenizer = load_model(config)
    
    print("Loading WikiText test data...")
    test_data = WikiTextDataSource(split="test")
    print(f"  Test samples: {len(test_data)}")
    
    calculator = PerplexityCalculator(tokenizer, max_length=max_window_size)
    
    # Convert action to layer ratios
    layer_ratios = get_layer_ratios(np.array(action))
    print(f"Layer ratios: {layer_ratios}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_fn = make_mask_fn(layer_ratios, device=device)
    
    print("Computing perplexity with pruning...")
    total_log_likelihood = 0.0
    total_tokens = 0
    
    start_time = time.time()
    prunable_llm.prune(mask_fn)
    end_pruning = time.time()
    print(f"Pruning time: {end_pruning - start_time:.2f} seconds")
    sparsity = prunable_llm.sparsity
    for item in tqdm(test_data, desc="Evaluating"):
        ll, tc = calculator.compute(prunable_llm, item)
        
        total_log_likelihood += ll
        total_tokens += tc
    
    s_unprune = time.time()
    prunable_llm.undo_prune()
    end_unpruning = time.time() 
    print(f"Unpruning time: {end_unpruning - s_unprune:.2f} seconds")
    end_time = time.time()
    eval_time = end_time - start_time
    print(f"Inference time with pruning: {eval_time - (end_pruning - start_time) - (end_unpruning - s_unprune):.2f} seconds")
    print(f"Total evaluation time: {eval_time:.2f} seconds")
    
    if total_tokens > 0:
        perplexity = np.exp(-total_log_likelihood / total_tokens)
        print(f"\nPruned Perplexity: {perplexity:.2f}")
        print(f"Sparsity: {sparsity:.2%}")
    else:
        print("No tokens to evaluate.")
    
    return perplexity, sparsity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaMA model on WikiText with or without pruning.")
    parser.add_argument(
        "--action",
        type=str,
        help="Comma-separated list of 32 floats for pruning action (e.g., '0.0,0.3,...'). If not provided, evaluates unpruned model."  # OLD: 8 floats
    )
    parser.add_argument(
        "--max_window_size",
        type=int,
        default=2048,
        help="Maximum window size for perplexity calculation e.g. --max_window_size 2048"
    )
    args = parser.parse_args()
    
    config = load_config()
    
    if args.action is None:
        print("Evaluating unpruned model...")
        unpruned_ppl = evaluate_unpruned(config, args.max_window_size)
    else:
        try:
            action = [float(x.strip()) for x in args.action.split(',')]
            if len(action) != 32:  # Changed from 8 to 32
                raise ValueError("Action must be exactly 32 floats.")  # Changed from 8 to 32
            print(f"Evaluating with pruning action: {action}")
            pruned_ppl, sparsity = evaluate_pruned(config, action, args.max_window_size)
        except ValueError as e:
            print(f"Error parsing action: {e}")
    
