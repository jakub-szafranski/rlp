import json

import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Literal
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from pruning import PrunableLLM
from rl.env import LLMPruningEnv
from rl.data_source import WikiTextDataSource, MMLUDataSource


def load_config(config_path: str = "config.yml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class MetricsCallback(BaseCallback):
    """Callback for logging metrics during training with tqdm progress bar."""
    
    def __init__(self, total_timesteps: int, task: str = "perplexity", log_interval: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.task = task
        self.log_interval = log_interval
        self.metrics = []  # perplexity or accuracy
        self.sparsities = []
        self.pbar = None
    
    def _on_training_start(self) -> None:
        """Initialize the progress bar at training start."""
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")
    
    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        
        infos = self.locals.get("infos", [])
        for info in infos:
            self.sparsities.append(info["sparsity"])
            if self.task == "perplexity":
                self.metrics.append(info.get("perplexity", 0))
            else:
                self.metrics.append(float(info.get("correct", False)))
        
        if self.n_calls % self.log_interval == 0 and self.metrics:
            recent_metrics = self.metrics[-self.log_interval:]
            recent_sparsity = self.sparsities[-self.log_interval:]
            
            if self.task == "perplexity":
                metric_str = f"PPL: {np.mean(recent_metrics):>8.2f}"
            else:
                metric_str = f"Acc: {np.mean(recent_metrics):>6.2%}"
            
            # Update progress bar postfix with metrics
            if self.pbar is not None:
                self.pbar.set_postfix_str(f"{metric_str} | Sparsity: {np.mean(recent_sparsity):.2%}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Close the progress bar at training end."""
        if self.pbar is not None:
            self.pbar.close()


def create_data_source(config: dict, split: str, max_samples: int | None = None):
    """Create WikiText or MMLU data source based on config."""
    data_conf = config["data"]
    dataset = data_conf["dataset"].lower()
    
    if dataset == "wikitext":
        return WikiTextDataSource(split=split, max_samples=max_samples)
    elif dataset == "mmlu":
        return MMLUDataSource(
            split=split,
            subjects=data_conf.get("mmlu_subjects"),
            max_samples=max_samples,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'wikitext' or 'mmlu'.")


def load_models(config: dict):
    """Load LLM, encoder, and cluster config."""
    model_conf = config["model"]
    encoder_conf = config["encoder"]
    train_conf = config["training"]
    device = train_conf.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading LLM: {model_conf['name']}...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_conf["name"],
        torch_dtype=getattr(torch, model_conf.get("dtype", "float16")),
        device_map=device,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(model_conf["name"])
    prunable_llm = PrunableLLM(llm)
    
    print(f"Loading encoder: {encoder_conf['name']}...")
    encoder = AutoModel.from_pretrained(encoder_conf["name"]).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_conf["name"])
    
    cluster_mapping_path = config["clusters"]["mapping_path"]
    print(f"Loading clusters from: {cluster_mapping_path}...")
    with open(cluster_mapping_path) as f:
        cluster_mapping = json.load(f)
    cluster_names = list(cluster_mapping.keys())
    print(f"  Loaded {len(cluster_names)} clusters")
    
    return {
        "prunable_llm": prunable_llm,
        "llm_tokenizer": llm_tokenizer,
        "encoder": encoder,
        "encoder_tokenizer": encoder_tokenizer,
        "cluster_names": cluster_names,
        "device": device,
    }


def create_env(config: dict, models: dict, data_source):
    """Create LLMPruningEnv."""
    env_conf = config["environment"]
    encoder_conf = config["encoder"]
    data_conf = config["data"]
    
    # Determine task type based on dataset
    dataset = data_conf["dataset"].lower()
    task = "correctness" if dataset == "mmlu" else "perplexity"
    
    return LLMPruningEnv(
        model=models["prunable_llm"],
        llm_tokenizer=models["llm_tokenizer"],
        encoder=models["encoder"],
        encoder_tokenizer=models["encoder_tokenizer"],
        data_source=data_source,
        cluster_names=models["cluster_names"],
        task=task,
        embed_dim=encoder_conf.get("embed_dim", 768),
        max_seq_len=env_conf.get("max_seq_len", 2048),
        quality_weight=env_conf.get("quality_weight", 0.7),
        device=models["device"],
    )


def train(config: dict):
    """Train PPO agent."""
    data_conf = config["data"]
    ppo_conf = config["ppo"]
    train_conf = config["training"]
    
    dataset = data_conf["dataset"].lower()
    train_split = "auxiliary_train" if dataset == "mmlu" else "train"
    
    models = load_models(config)
    
    print(f"Loading data source: {dataset} ({train_split})...")
    train_data = create_data_source(
        config, 
        split=train_split, 
        max_samples=data_conf.get("train_samples")
    )
    
    print("Creating environment...")
    env = create_env(config, models, train_data)
    
    print("Creating PPO agent...")
    policy_kwargs = dict(
        net_arch=dict(
            pi=ppo_conf["pi_layers"],
            vf=ppo_conf["vf_layers"],
        ),
        activation_fn=torch.nn.GELU,
    )
    
    agent = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_conf.get("learning_rate", 3e-4),
        n_steps=ppo_conf.get("n_steps", 64),
        batch_size=ppo_conf.get("batch_size", 32),
        n_epochs=ppo_conf.get("n_epochs", 4),
        gamma=ppo_conf.get("gamma", 0.0),  # 0 for contextual bandits
        ent_coef=ppo_conf.get("ent_coef", 0.01),
        clip_range=ppo_conf.get("clip_range", 0.2),
        max_grad_norm=ppo_conf.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=train_conf.get("verbose", 1),
        seed=config.get("random_state", 42),
    )
    
    print(f"\nStarting training for {train_conf['total_timesteps']} steps...")
    print("=" * 60)
    
    task = "correctness" if dataset == "mmlu" else "perplexity"
    callback = MetricsCallback(
        total_timesteps=train_conf["total_timesteps"],
        task=task, 
        log_interval=train_conf.get("log_interval", 100)
    )
    agent.learn(total_timesteps=train_conf["total_timesteps"], callback=callback)
    
    save_path = train_conf.get("save_path", "ppo_pruning_agent")
    agent.save(save_path)
    print(f"\nModel saved to: {save_path}")


def evaluate(config: dict):
    """Evaluate trained PPO agent on test set."""
    data_conf = config["data"]
    train_conf = config["training"]
    eval_conf = config["evaluation"]
    
    dataset = data_conf["dataset"].lower()
    task = "correctness" if dataset == "mmlu" else "perplexity"
    test_split = "test"
    
    models = load_models(config)
    
    print(f"Loading test data: {dataset} ({test_split})...")
    test_data = create_data_source(config, split=test_split, max_samples=None)
    print(f"  Test samples: {len(test_data)}")
    
    print("Creating environment...")
    env = create_env(config, models, test_data)
    
    load_path = train_conf.get("save_path", "ppo_pruning_agent")
    print(f"Loading agent from: {load_path}...")
    agent = PPO.load(load_path, env=env)
    
    print(f"\nEvaluating on {len(test_data)} test samples...")
    print("=" * 60)
    
    deterministic = eval_conf.get("deterministic", True)
    
    results = {"sparsity": [], "reward": [], "clusters_pruned": []}
    if task == "perplexity":
        results["perplexity"] = []
    else:
        results["correct"] = []
    
    obs, _ = env.reset()
    for _ in tqdm(range(len(test_data)), desc="Evaluating"):
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        
        results["sparsity"].append(info["sparsity"])
        results["reward"].append(reward)
        results["clusters_pruned"].append(info["num_clusters_pruned"])
        
        if task == "perplexity":
            results["perplexity"].append(info["perplexity"])
        else:
            results["correct"].append(info["correct"])
        
        if terminated:
            obs, _ = env.reset()
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    if task == "perplexity":
        print(f"Mean Perplexity:      {np.mean(results['perplexity']):>10.2f} ± {np.std(results['perplexity']):>8.2f}")
    else:
        accuracy = np.mean(results["correct"])
        print(f"Accuracy:             {accuracy:>10.2%}")
    
    print(f"Mean Sparsity:        {np.mean(results['sparsity']):>10.2%} ± {np.std(results['sparsity']):>8.2%}")
    print(f"Mean Reward:          {np.mean(results['reward']):>10.4f} ± {np.std(results['reward']):>8.4f}")
    print(f"Mean Clusters Pruned: {np.mean(results['clusters_pruned']):>10.1f} / {len(models['cluster_names'])}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM pruning training or evaluation.")
    parser.add_argument("run_mode", choices=["train", "eval"], help="Mode to run: train or eval")
    args = parser.parse_args()
    run_mode = args.run_mode

    with open('config.yml') as f:
        config = yaml.safe_load(f)
    
    if run_mode == "train":
        train(config)
    elif run_mode == "eval":
        evaluate(config)
    else:
        raise ValueError(f"Unknown run mode: {run_mode}. Use 'train' or 'eval'.")