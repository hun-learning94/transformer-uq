import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuration
# ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

d = 16           # Fixed dimension d
sigma_sq = 0.2   # Given sigma^2
n_values = torch.linspace(100, 1000, steps=20).int().tolist() # Sample sizes n
# n_values = [64 * i for i in range(1, 16)]
trials = 100       # Average over a few trials for smoother plots

# 2. Kernel Functions
# -------------------
def linear_kernel(X):
    # X: (n, d)
    # G = X * X^T
    return torch.matmul(X, X.T)

def rbf_kernel(X, sigma=1.0):
    # X: (n, d)
    # G_ij = exp(-||x_i - x_j||^2 / 2)
    # PyTorch cdist computes p-norm distance. We need squared Euclidean.
    dists = torch.cdist(X, X, p=2).pow(2)
    return torch.exp(-dists / 2)

# 3. Simulation Loop
# ------------------
results_linear = []
results_rbf = []

print("Starting simulation...")

for n in n_values:
    val_linear_accum = 0.0
    val_rbf_accum = 0.0
    
    for _ in range(trials):
        # Generate Data: x_i ~ N(0, I_d / d)
        # Variance is 1/d, so std is 1/sqrt(d)
        X = torch.randn(n, d, device=device) / np.sqrt(d)
        
        # Linear Kernel
        G_lin = linear_kernel(X)
        # Use eigvalsh because G is symmetric (faster/stable)
        # We only need the largest eigenvalue (last one in the sorted list)
        lambda_1_lin = torch.linalg.eigvalsh(G_lin)[-1]
        val_linear_accum += 1.0 / (lambda_1_lin + sigma_sq)
        
        # RBF Kernel
        G_rbf = rbf_kernel(X)
        lambda_1_rbf = torch.linalg.eigvalsh(G_rbf)[-1]
        val_rbf_accum += 1.0 / (lambda_1_rbf + sigma_sq)
    
    # Average over trials
    results_linear.append(val_linear_accum.item() / trials)
    results_rbf.append(val_rbf_accum.item() / trials)

print("Simulation complete.")

# 4. Plotting
# -----------
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))
ns = np.array(n_values)

# Helper for reference line C/n
def fit_inv_n(n_data, y_data):
    # Fit y = C * (1/n) -> C = y * n
    # We take the mean C from the data to align the curve
    C = np.mean(np.array(n_data) * np.array(y_data))
    return C / n_data

# Plot 1: Linear Kernel
axes[0].plot(ns, results_linear, 'o-', label='Simulation', color='blue', markersize=4)
axes[0].plot(ns, fit_inv_n(ns, results_linear), '--', label=r'Ref $\Theta(1/n)$', color='red', alpha=0.7)
axes[0].set_title(r"Linear Kernel")
axes[0].set_xlabel("Sample Size ($n$)")
axes[0].set_ylabel(r"$\eta$ Upper Bound")
axes[0].legend()
axes[0].grid(True, which='both', linestyle='--', alpha=0.6)

# Plot 2: RBF Kernel
axes[1].plot(ns, results_rbf, 'o-', label='Simulation', color='green', markersize=4)
axes[1].plot(ns, fit_inv_n(ns, results_rbf), '--', label=r'Ref $\Theta(1/n)$', color='red', alpha=0.7)
axes[1].set_title(r"RBF Kernel")
axes[1].set_xlabel("Sample Size ($n$)")
axes[0].set_ylabel(r"Admissible $\eta$")
axes[1].legend()
axes[1].grid(True, which='both', linestyle='--', alpha=0.6)

plt.tight_layout()
# plt.show()
plt.savefig("figs/figure4.pdf", dpi=300, bbox_inches="tight")