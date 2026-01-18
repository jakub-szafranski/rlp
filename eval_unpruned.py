import yaml
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time

from pruning import PrunableLLM
from pruning.create_pruning_mask import make_mask_fn
from rl.data_source import WikiTextDataSource, MMLUDataSource
from rl.metrics import PerplexityCalculator, MMLULoglikelihoodCalculator

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


def evaluate_unpruned(config: dict, task: str, max_window_size: int):
    """Evaluate unpruned model on test set."""
    prunable_llm, tokenizer = load_model(config)
    
    if task == "mmlu":
        print("Loading MMLU test data...")
        test_data = MMLUDataSource(split="test", subjects=config['data']['mmlu_subjects'])
        print(f"  Test samples: {len(test_data)}")
        calculator = MMLULoglikelihoodCalculator(tokenizer)
        
        print("Computing accuracy...")
        correct = 0
        
        start_time = time.time()
        for item in tqdm(test_data, desc="Evaluating"):
            diff = calculator.compute(prunable_llm, item)
            if diff > 0:
                correct += 1
        end_time = time.time()
        
        eval_time = end_time - start_time
        print(f"Evaluation time: {eval_time:.2f} seconds")
        
        accuracy = correct / len(test_data)
        print(f"\nUnpruned Accuracy: {accuracy:.2%}")
        return accuracy
    else:
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
            perplexity = float('inf')
        
        return perplexity


def evaluate_pruned(config: dict, action: list[float], max_window_size: int, task: str):
    """Evaluate pruned model on test set with given action."""
    prunable_llm, tokenizer = load_model(config)
    
    if task == "mmlu":
        print("Loading MMLU test data...")
        test_data = MMLUDataSource(split="test", subjects=config['data']['mmlu_subjects'])
        print(f"  Test samples: {len(test_data)}")
        calculator = MMLULoglikelihoodCalculator(tokenizer)
    else:
        print("Loading WikiText test data...")
        test_data = WikiTextDataSource(split="test")
        print(f"  Test samples: {len(test_data)}")
        
        calculator = PerplexityCalculator(tokenizer, max_length=max_window_size)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_fn = make_mask_fn(action, device=device)
    
    print("Computing metric with pruning...")
    start_time = time.time()
    prunable_llm.prune(mask_fn)
    end_pruning = time.time()
    print(f"Pruning time: {end_pruning - start_time:.2f} seconds")
    sparsity = prunable_llm.sparsity
    
    if task == "mmlu":
        correct = 0
        for item in tqdm(test_data, desc="Evaluating"):
            diff = calculator.compute(prunable_llm, item)
            if diff > 0:
                correct += 1
        
        s_unprune = time.time()
        prunable_llm.undo_prune()
        end_unpruning = time.time() 
        print(f"Unpruning time: {end_unpruning - s_unprune:.2f} seconds")
        end_time = time.time()
        eval_time = end_time - start_time
        print(f"Inference time with pruning: {eval_time - (end_pruning - start_time) - (end_unpruning - s_unprune):.2f} seconds")
        print(f"Total evaluation time: {eval_time:.2f} seconds")
        
        accuracy = correct / len(test_data)
        print(f"\nPruned Accuracy: {accuracy:.2%}")
        print(f"Sparsity: {sparsity:.2%}")
        return accuracy, sparsity
    else:
        total_log_likelihood = 0.0
        total_tokens = 0
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
            perplexity = float('inf')
        
        return perplexity, sparsity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaMA model on WikiText with or without pruning.")
    parser.add_argument(
        "--action",
        type=str,
        help="Comma-separated list of 8 floats for pruning action (e.g., '0.0,0.3,...'). If not provided, evaluates unpruned model."
    )
    parser.add_argument(
        "--max_window_size",
        type=int,
        default=2048,
        help="Maximum window size for perplexity calculation e.g. --max_window_size 2048"
    )    
    parser.add_argument(
        "--task",
        type=str,
        default="wikitext",
        help="wikitext or mmlu"
    )
    args = parser.parse_args()
    
    config = load_config()
    
    if args.action is None:
        print("Evaluating unpruned model...")
        result = evaluate_unpruned(config, args.task, args.max_window_size)
    else:
        try:
            action = [float(x.strip()) for x in args.action.split(',')]
            print(f"Evaluating with pruning action: {action}")
            result, sparsity = evaluate_pruned(config, action, args.max_window_size, args.task)
        except ValueError as e:
            print(f"Error parsing action: {e}")
    
