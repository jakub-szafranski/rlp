import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rl.reward import PerplexityReward

# Create reward calculator
reward_calc = PerplexityReward(quality_weight=0.5, ppl_sensitivity=1.5, sparsity_sensitivity=3.0)

# Assume baseline perplexity
baseline_ppl = 10.0

# Vary perplexity ratio and sparsity
ratios = np.linspace(0.0, 1.5, 50)
sparsities = np.linspace(0, 1, 50)

RR, SS = np.meshgrid(ratios, sparsities)
rewards = np.zeros_like(RR)

for i in range(RR.shape[0]):
    for j in range(RR.shape[1]):
        pruned_ppl = RR[i, j] * baseline_ppl
        rewards[i, j] = reward_calc.compute_reward(pruned_ppl, baseline_ppl, SS[i, j])

# Plot 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(RR, SS, rewards, cmap='viridis', edgecolor='none')
ax.set_xlabel('Perplexity Ratio')
ax.set_ylabel('Sparsity')
ax.set_zlabel('Reward')
ax.set_title('Reward Surface for Different Perplexity Ratios and Sparsity Pairs')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()

# Alternative visualization: Contour plot
fig2, ax2 = plt.subplots(figsize=(10, 8))
contour = ax2.contourf(RR, SS, rewards, levels=20, cmap='viridis')
ax2.set_xlabel('Perplexity Ratio')
ax2.set_ylabel('Sparsity')
ax2.set_title('Reward Contour Plot')
fig2.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
plt.show()