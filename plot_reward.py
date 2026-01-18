from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from rl.reward import PerplexityReward, CorrectnessReward

reward = "mmlu" # mmlu or ppl
if reward == "ppl":
    ppl_reward = PerplexityReward()

    baseline_ppl = 10.0
    for ppl_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for sparsity in [0.0, 0.1, 0.2, 0.29, 0.4]:
            pruned_ppl = baseline_ppl * (1 + ppl_ratio)
            reward = ppl_reward.compute_reward(pruned_ppl, baseline_ppl, sparsity)
            print(f"PPL Ratio: {ppl_ratio:.2f}, Sparsity: {sparsity:.2f} => Reward: {reward:.3f}")

    # Plotting the reward surface
    import matplotlib.pyplot as plt

    ppl_ratios = np.linspace(0.0, 0.6, 50)
    sparsities = np.linspace(0.0, 0.5, 50)
    PR, SP = np.meshgrid(ppl_ratios, sparsities)
    rewards = np.zeros_like(PR)

    for i in range(PR.shape[0]):
        for j in range(PR.shape[1]):
            pruned_ppl = baseline_ppl * (1 + PR[i, j])
            rewards[i, j] = ppl_reward.compute_reward(pruned_ppl, baseline_ppl, SP[i, j])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(PR, SP, rewards, cmap='viridis', edgecolor='none')
    ax.set_xlabel('PPL Ratio')
    ax.set_ylabel('Sparsity')
    ax.set_zlabel('Reward')
    ax.set_title('Reward Surface for Different PPL Ratios and Sparsity Pairs')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

elif reward == "mmlu":
    mmlu_reward = CorrectnessReward()

    for margin in [-2, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        for sparsity in [0.0, 0.1, 0.2, 0.29, 0.4]:
            reward = mmlu_reward.compute_reward(margin, sparsity)
            print(f"Margin: {margin:.2f}, Sparsity: {sparsity:.2f} => Reward: {reward:.3f}")
    
    # Plotting the reward surface
    import matplotlib.pyplot as plt
    margins = np.linspace(-2.0, 2.0, 50)
    sparsities = np.linspace(0.0, 0.5, 50)
    MG, SP = np.meshgrid(margins, sparsities)
    rewards = np.zeros_like(MG)
    for i in range(MG.shape[0]):
        for j in range(MG.shape[1]):
            rewards[i, j] = mmlu_reward.compute_reward(MG[i, j], SP[i, j])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(MG, SP, rewards, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Margin')
    ax.set_ylabel('Sparsity')
    ax.set_zlabel('Reward')
    ax.set_title('Reward Surface for Different Margins and Sparsity Pairs')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()