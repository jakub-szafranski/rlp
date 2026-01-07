# Copilot Instructions for this Repo

## Project overview
- **Goal**: Train a PPO agent to learn pruning policies for a large language model (LLaMA 3.1 8B Instruct) based on precomputed neuron clusters, trading off quality (perplexity/accuracy) against sparsity.
- **Core idea**: Treat pruning as a contextual bandit RL problem. Each episode:
  - Sample a text example from WikiText (perplexity) or MMLU (multiple-choice correctness).
  - Encode the text into a fixed-size embedding (ModernBERT).
  - The PPO agent outputs a binary mask over clusters (which clusters to prune).
  - The environment prunes the model according to this mask, evaluates quality, restores the model, and returns a reward combining quality and sparsity.

## High-level architecture
- `main.py`
  - Entry point for running **training** (`train`) and **evaluation** (`eval`).
  - Loads configuration from `config.yml`.
  - **Key functions**:
    - `load_config(config_path)` – YAML config loader.
    - `create_data_source(config, split, max_samples)` – creates either `WikiTextDataSource` or `MMLUDataSource` based on `config['data']['dataset']`.
    - `load_models(config)` – loads:
      - HF causal LLM (`AutoModelForCausalLM`) and tokenizer.
      - HF encoder (`AutoModel`) and tokenizer.
      - Wraps LLM in `PrunableLLM`.
      - Loads cluster mapping JSON to get cluster names.
    - `create_env(config, models, data_source)` – builds `LLMPruningEnv` with task type inferred from dataset.
    - `train(config)` – wires everything together and runs PPO training with `stable_baselines3.PPO`.
    - `evaluate(config)` – loads a saved agent and runs evaluation loop over test split.
  - Uses `MetricsCallback` (tqdm-based) to log progress, sparsity and metric (perplexity/accuracy) during training.

- `pruning/`
  - `PrunableLLM` (in `prunable_llm.py`):
    - Lightweight wrapper around a HF `PreTrainedModel`.
    - For external callers, behaves like the underlying model (`__getattr__`, `__call__`, `forward`, `generate`, `.config`).
    - Adds:
      - `.prune(mask_fn, storage='cpu'|'gpu', cast_dtype)` – prunes all layers using a provided mask function (per-layer boolean tensor) and stores removed weights in an `UndoPack`.
      - `.undo_prune(device=None)` – restores original weights from the `UndoPack` and clears it.
      - `.sparsity` – property exposing current fraction of pruned neurons.
      - `.is_pruned` – whether model is currently pruned.

  - `create_pruning_mask.py`:
    - Loads precomputed masks and cluster↔layer mapping from JSON files.
    - `get_pruning_mask(clusters_to_prune, layer)` – returns a NumPy 1D mask for a given layer based on selected clusters.
    - Used indirectly via RL utilities to convert cluster actions into per-layer neuron masks.

  - `utils.py`:
    - Defines the low-level pruning machinery:
      - `UndoPack`, `LayerUndoEntry`, and `Dims` (Pydantic models) store removed weights and indices per layer.
      - `prune_with_undo_pack(model, mask_fn, storage, cast_dtype)` – iterates over transformer MLP layers, applies masks to gate/up/down projections, updates shapes, and stores removed slices + metadata.
      - `unprune_from_undo_pack(model, pack, device)` – reconstructs full weight matrices using stored slices and indices, then cleans up memory.

- `rl/`
  - `env.py` – `LLMPruningEnv` (Gymnasium env):
    - **Observation**: embedding of current text example from encoder.
    - **Action**: `MultiBinary(num_clusters)` – binary mask over cluster names.
    - **Reward**: computed from either perplexity change (WikiText) or correctness (MMLU) plus sparsity.
    - Flow in `step(action)`:
      1. Optionally compute baseline perplexity on the unpruned model (for WikiText).
      2. Build a `mask_fn` from the binary action using `MaskFunctionAdapter`.
      3. Call `model.prune(mask_fn, storage='gpu')`.
      4. Compute metric using `self.metric_calculator.compute`.
      5. Read sparsity from `model.sparsity`.
      6. Call `model.undo_prune()` to restore full model.
      7. Compute reward via `self.reward_calculator.compute_reward(...)`.
      8. Sample next item and return its embedding.

  - `data_source.py`:
    - Abstract `DataSource` base class with `__iter__` and `__len__`.
    - `WikiTextDataSource`:
      - Uses `datasets.load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1")`.
      - Detokenizes text with `wikitext_detokenizer` to match lm-eval.
      - Each yielded item: `{'text', 'original_text', 'word_count', 'type': 'wikitext'}`.
    - `MMLUDataSource`:
      - Uses `datasets.load_dataset("cais/mmlu", "all")`.
      - Optional subject filtering and max sample limiting.
      - Formats questions as "Question... A. ... B. ... C. ... D. ... Answer:".
      - Each item includes text, choices, answer index/letter, subject.

  - `metrics.py`:
    - Abstract `MetricCalculator`.
    - `PerplexityCalculator`:
      - Computes **word-level** perplexity on WikiText (non-overlapping windows, EOS prepended), similar to lm-eval.
    - `MMLULoglikelihoodCalculator`:
      - Computes per-choice loglikelihood of full choice text conditioned on the prompt.
      - Normalizes by token count and picks argmax; returns a bool for correctness.

  - `reward.py`:
    - Abstract `RewardCalculator` with `quality_weight` ∈ [0,1].
    - `PerplexityReward`:
      - Combines sparsity and perplexity ratio via `tanh`, returns value in [-1,1].
    - `CorrectnessReward`:
      - Combines ±1 correctness signal with sparsity-based reward.

  - `utils.py` (RL):
    - `MaskFunctionAdapter` (not shown here, but used by the env) converts a cluster-level binary action into the `mask_fn(layer_idx) -> bool tensor` expected by `PrunableLLM.prune`.

- `grouping_statistics/` & JSONs
  - Hold clustering/grouping artifacts (e.g., `cluster_layer_mapping_up_2l_16c.json`, `cluster_masks_up_2l_16c.json`).
  - These describe which neurons (per layer) belong to which high-level clusters.

- `playground/`
  - Jupyter notebooks and experiments (perplexity measurements, spectral clustering, RL tests, etc.).
  - Not part of the main training pipeline, but useful for ad‑hoc analysis.

## How to run
- Training:
  - Entry: `main.py`, function `train`.
  - CLI usage:
    - `python main.py train`
  - Uses `config.yml` to configure model name, encoder, data, PPO, etc.

- Evaluation:
  - Entry: `main.py`, function `evaluate`.
  - CLI usage:
    - `python main.py eval`
  - Expects a trained PPO checkpoint at `training.save_path` from `config.yml`.

## Important conventions / expectations for Copilot

### General style
- Use **type hints** consistently, including return types.
- Prefer **pure functions** where practical; keep side effects (file I/O, logging) localized in high-level orchestration code (`main.py`).
- Use `torch.no_grad()` for evaluation / inference paths and pruning operations to avoid unnecessary gradient tracking.
- Keep imports explicit and grouped (standard library, third-party, local modules).

### Model and pruning assumptions
- The prunable LLM is assumed to be a transformer with a `.model.layers` list and each layer having an `mlp` with `gate_proj`, `up_proj`, and `down_proj` linear layers.
- Any new code that manipulates weights should respect this structure or be explicitly adapted.
- `mask_fn(layer_idx)` should always return a **1D boolean tensor** of length equal to the MLP intermediate dimension (`inter`), where `True` = keep neuron, `False` = prune.
- After calling `.prune(...)`, always ensure `.undo_prune()` is eventually called before reusing the model for another operation, otherwise subsequent steps may silently operate on a partially pruned model.

### RL environment and data
- The environment is designed as a **single-step contextual bandit**:
  - `reset()` gives an embedding for a sampled item.
  - `step(action)` prunes, evaluates, restores, computes reward, then moves to the next sample and returns its embedding.
- When adding new tasks / metrics:
  - Implement a new `MetricCalculator` subclass in `rl/metrics.py`.
  - Implement a matching `RewardCalculator` subclass in `rl/reward.py`.
  - Extend `LLMPruningEnv` and `create_env` in `main.py` to select these based on `config.yml`.

### Configuration and extensibility
- All experiment-level choices should go through `config.yml` where possible:
  - Model/encoder names and dtypes.
  - Dataset selection and sample counts.
  - PPO hyperparameters and network architectures.
  - Environment hyperparameters (max seq length, quality vs sparsity weight).
- When adding new options, prefer to:
  - Add them to `config.yml` with comments.
  - Access them via `config[...]` in `main.py` or the relevant constructor.

### Error handling and logging
- Fail fast on configuration errors (e.g., unknown dataset names, missing files) using `ValueError` with a clear message.
- Use `print` for high-level progress / checkpoints (loading models, data sizes, start/end of training/eval) and tqdm for long-running loops.

### What to avoid
- Do **not** silently change the semantics of sparsity (e.g., from fraction of pruned neurons to kept neurons) – if needed, introduce a new field instead.
- Do not hardcode model- or dataset-specific paths; use configuration or relative paths within the repo.
- Avoid adding heavyweight dependencies unless clearly justified and necessary.

### Personal preferences
- I like clean code, short code, fast and vectorized code.
- to run things use: python3 <file_name>.py
- always do at the beggining conda activate llm_pruning
