import json

import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Literal
from stable_baselines3 import SAC
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
    
    def __init__(self, total_timesteps: int, task: str = "perplexity", log_interval: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.task = task
        self.log_interval = log_interval
        self.sparsities = []
        # Recent rewards collected from trainer locals (if available)
        self.rewards = []
        # For perplexity: aggregate log-likelihood and word counts for proper PPL calculation
        self.log_likelihoods = []
        self.baseline_log_likelihoods = []
        self.word_counts = []
        # For accuracy
        self.accuracies = []
        self.pbar = None
        # Permanent history of logged aggregates (not overwritten by tqdm)
        # Each entry: dict with step, metric values and sparsity mean
        self.history = []
        # Count of info entries appended (one per env step/info dict)
        self.info_count = 0
    
    def _on_training_start(self) -> None:
        """Initialize the progress bar at training start."""
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")
    
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        # Number of environment info dicts received in this callback
        n_infos = len(infos) if infos is not None else 1
        if self.pbar is not None:
            # Update progress bar by number of env steps processed
            self.pbar.update(n_infos)

        for info in infos:
            self.sparsities.append(info["sparsity"])
            if self.task == "perplexity":
                self.log_likelihoods.append(info.get("log_likelihood", 0.0))
                self.baseline_log_likelihoods.append(info.get("baseline_log_likelihood", 0.0))
                self.word_counts.append(info.get("word_count", 0))
            else:
                self.accuracies.append(float(info.get("correct", False)))

        # Try to collect rewards from trainer locals (stable-baselines3 provides
        # a `rewards` entry containing the reward(s) for the recent step(s)).
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            # Normalize to a Python list of floats
            try:
                if isinstance(rewards, np.ndarray):
                    rewards_list = rewards.tolist()
                elif isinstance(rewards, (float, int)):
                    rewards_list = [float(rewards)]
                else:
                    rewards_list = list(rewards)
            except Exception:
                rewards_list = [float(rewards)]

            # Align number of reward entries with number of infos (if mismatched,
            # only take the common prefix)
            if len(rewards_list) != n_infos:
                m = min(len(rewards_list), n_infos)
                rewards_list = rewards_list[:m]

            self.rewards.extend([float(r) for r in rewards_list])

        # Increment total number of appended info entries and trigger
        # logging when we've accumulated `log_interval` env steps.
        self.info_count += len(infos)

        if self.info_count % self.log_interval == 0:
            recent_sparsity = self.sparsities[-self.log_interval:]
            
            if self.task == "perplexity":
                # Aggregate log-likelihoods and word counts for proper perplexity
                recent_ll = self.log_likelihoods[-self.log_interval:]
                recent_baseline_ll = self.baseline_log_likelihoods[-self.log_interval:]
                recent_wc = self.word_counts[-self.log_interval:]
                total_wc = sum(recent_wc)
                if total_wc > 0:
                    pruned_ppl = np.exp(-sum(recent_ll) / total_wc)
                    baseline_ppl = np.exp(-sum(recent_baseline_ll) / total_wc)
                    metric_str = f"PPL: {pruned_ppl:>6.2f} (base: {baseline_ppl:.2f})"
                else:
                    metric_str = "PPL: N/A"
            else:
                recent_acc = self.accuracies[-self.log_interval:]
                metric_str = f"Acc: {np.mean(recent_acc):>6.2%}"
            # Recent mean reward, if available
            recent_rewards = self.rewards[-self.log_interval:] if len(self.rewards) >= 1 else []
            if len(recent_rewards) > 0:
                mean_reward = float(np.mean(recent_rewards))
                reward_str = f" | AvgReward: {mean_reward:.4f}"
            else:
                mean_reward = None
                reward_str = ""
            
            # Keep the tqdm postfix compact (sparsity only) so the detailed
            # metric values are not overwritten in place. Also emit a permanent
            # log line and store to history so past values remain available.
            if self.pbar is not None:
                self.pbar.set_postfix_str(f"Sparsity: {np.mean(recent_sparsity):.2%}")

            # Permanent log line (prints to stdout and remains in terminal history)
            print(f"[{self.n_calls}] {metric_str} | Sparsity: {np.mean(recent_sparsity):.2%}{reward_str}")

            # Save aggregate snapshot to history
            self.history.append(
                    {
                        "step": int(self.n_calls),
                        "metric": metric_str,
                        "sparsity": float(np.mean(recent_sparsity)),
                        "reward": (float(mean_reward) if mean_reward is not None else None),
                    }
            )
        
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
    """Load LLM, encoder."""
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
    
    return {
        "prunable_llm": prunable_llm,
        "llm_tokenizer": llm_tokenizer,
        "encoder": encoder,
        "encoder_tokenizer": encoder_tokenizer,
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
        task=task,
        embed_dim=encoder_conf.get("embed_dim", 768),
        max_seq_len=env_conf.get("max_seq_len", 2048),
        quality_weight=env_conf.get("quality_weight", 0.7),
        baseline_perplexity=env_conf["baseline_perplexity"],
        device=models["device"],
    )


def train(config: dict):
    """Train SAC agent."""
    data_conf = config["data"]
    sac_conf = config["sac"]
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
    
    print("Creating SAC agent...")
    policy_kwargs = dict(
        net_arch=dict(
            pi=sac_conf["pi_layers"],
            qf=sac_conf["qf_layers"],
        ),
        activation_fn=torch.nn.GELU,
    )
    
    agent = SAC(
        "MlpPolicy",
        env,
        learning_rate=sac_conf.get("learning_rate", 3e-4),
        buffer_size=sac_conf.get("buffer_size", 100000),
        batch_size=sac_conf.get("batch_size", 256),
        tau=sac_conf.get("tau", 0.005),
        gamma=sac_conf.get("gamma", 0.99),
        train_freq=sac_conf.get("train_freq", 1),
        gradient_steps=sac_conf.get("gradient_steps", 1),
        ent_coef=sac_conf.get("ent_coef", "auto"),
        target_update_interval=sac_conf.get("target_update_interval", 1),
        policy_kwargs=policy_kwargs,
        verbose=train_conf.get("verbose", 1),
        seed=config.get("random_state", 42),
    )
    
    # Initialize action head to favor NOT pruning (lower initial sparsity)
    # This helps exploration across different sparsity levels instead of always 50%
    action_bias = sac_conf.get("action_bias", 0.1)
    if action_bias != 0.0:
        with torch.no_grad():
            action_net = agent.policy.action_net
            # Scale down weights so bias dominates initial output
            action_net.weight.data *= 0.5
            action_net.bias.fill_(action_bias)
            print(f"  Initialized action bias to {action_bias} (initial mean fraction: {action_bias})")


    
    print(f"\nStarting training for {train_conf['total_timesteps']} steps...")
    print("=" * 60)
    
    task = "correctness" if dataset == "mmlu" else "perplexity"
    callback = MetricsCallback(
        total_timesteps=train_conf["total_timesteps"],
        task=task, 
        log_interval=train_conf.get("log_interval", 50)
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
    agent = SAC.load(load_path, env=env)
    
    print(f"\nEvaluating on {len(test_data)} test samples...")
    print("=" * 60)
    
    deterministic = eval_conf.get("deterministic", True)
    
    results = {"sparsity": [], "reward": [], "mean_fraction_pruned": []}
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
        results["mean_fraction_pruned"].append(info["mean_fraction_pruned"])
        
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
    print(f"Mean Fraction Pruned: {np.mean(results['mean_fraction_pruned']):>10.2%} ± {np.std(results['mean_fraction_pruned']):>8.2%}")
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