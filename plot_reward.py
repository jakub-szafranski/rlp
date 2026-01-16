from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from rl.reward import PerplexityReward

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