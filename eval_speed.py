"""
Speed evaluation for PrunableLLM with and without agent-based pruning.

Evaluates generation speed for 10, 50, 100 new tokens on:
1. Unpruned model
2. Pruned model using trained SAC agent

Repeats each test 3 times (skipping first for caching), reports mean and std time.
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import numpy as np
import yaml
import time
from stable_baselines3 import SAC
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

from pruning import PrunableLLM, make_mask_fn


def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_models(config):
    model_conf = config["model"]
    encoder_conf = config["encoder"]
    train_conf = config["training"]
    env_conf = config["environment"]
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

    return prunable_llm, llm_tokenizer, encoder, encoder_tokenizer, agent, device, env_conf["pruning_type"]


@torch.no_grad()
def encode_text(text, encoder, encoder_tokenizer, device):
    inputs = encoder_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    outputs = encoder(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embedding.cpu().numpy().astype(np.float32)


def evaluate_speed(prunable_llm, tokenizer, encoder, encoder_tokenizer, agent, device, pruning_type, prompt="Once upon a time"):
    print(f"Using prompt: '{prompt}'")

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    num_tokens_list = [10, 50, 100, 250, 500]

    # Unpruned evaluation
    print("\n=== Unpruned Model ===")
    for num_tokens in num_tokens_list:
        times = []
        for i in range(4):  # 1 warm-up + 3 measured
            start_time = time.time()
            output = prunable_llm.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            end_time = time.time()
            if i > 0:  # Skip first
                times.append(end_time - start_time)
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"  {num_tokens} tokens: {mean_time:.3f}s ± {std_time:.3f}s")
        if num_tokens == num_tokens_list[-1]:
            print(f"Generated text: {tokenizer.decode(output[0], skip_special_tokens=True)}")
            print("")

    # Pruned evaluation
    print("\n=== Pruned Model (with agent) ===")
    observation = encode_text(prompt, encoder, encoder_tokenizer, device)
    action, _ = agent.predict(observation, deterministic=True)
    mask_fn = make_mask_fn(list(action), device, pruning_type=pruning_type)
    prunable_llm.prune(mask_fn)
    print(f"Sparsity after pruning: {prunable_llm.sparsity:.2%}")

    try:
        for num_tokens in num_tokens_list:
            times = []
            for i in range(4):  # 1 warm-up + 3 measured
                start_time = time.time()
                output = prunable_llm.generate(
                    input_ids,
                    max_new_tokens=num_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                end_time = time.time()
                if i > 0:  # Skip first
                    times.append(end_time - start_time)
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  {num_tokens} tokens: {mean_time:.3f}s ± {std_time:.3f}s")
            if num_tokens == num_tokens_list[-1]:
                print(f"Generated text: {tokenizer.decode(output[0], skip_special_tokens=True)}")
                print("")

    finally:
        prunable_llm.undo_prune()


if __name__ == "__main__":
    config = load_config()
    prunable_llm, tokenizer, encoder, encoder_tokenizer, agent, device, pruning_type = create_models(config)

    prompt = "Write a short story about a brave knight who saves a village from a dragon.\n\nLong ago, in a land far away, "

    evaluate_speed(prunable_llm, tokenizer, encoder, encoder_tokenizer, agent, device, pruning_type, prompt)