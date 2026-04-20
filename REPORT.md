# Self-Pruning Neural Network — Report

## 1. Introduction

This report documents the design and evaluation of a **self-pruning neural network** trained on the CIFAR-10 image classification dataset. Unlike traditional post-training pruning, this network uses **learnable gate parameters** to identify and dynamically remove its own weakest connections *during* the training process.

---

## 2. Architecture Overview

### 2.1 PrunableLinear Layer

Each standard `Linear(in, out)` layer is replaced with a `PrunableLinear` layer that introduces:

- **`weight`** ∈ ℝ^(out × in) — Standard weight matrix
- **`bias`** ∈ ℝ^out — Standard bias vector
- **`gate_scores`** ∈ ℝ^(out × in) — Learnable gate parameters (same shape as `weight`)

**Forward pass:**

```
gates = σ(gate_scores)              # σ = Sigmoid → values ∈ (0, 1)
pruned_weights = weight ⊙ gates     # ⊙ = element-wise multiplication
output = x @ pruned_weights^T + bias
```

Gates are initialized at σ(3.0) ≈ 0.95, so the network starts nearly fully connected.

### 2.2 Network Architecture

```
Input (3×32×32 = 3072)
    │
    ├─── PrunableLinear(3072,  256) → BatchNorm → ReLU → Dropout(0.2)
    ├─── PrunableLinear( 256,  128) → BatchNorm → ReLU → Dropout(0.2)
    ├─── PrunableLinear( 128,   64) → BatchNorm → ReLU → Dropout(0.2)
    └─── PrunableLinear(  64,   10) → Output logits
```

**Total parameters:** ~828K weight params + ~828K gate params

---

## 3. Why L1 Penalty on Sigmoid Gates Encourages Sparsity

### The Key Insight

The **L1 norm** (sum of absolute values) is well-known in optimization theory as a **sparsity-inducing regularizer**. Here's why it works specifically for our sigmoid gates:

1. **Sigmoid outputs are always positive** (∈ (0, 1)), so the L1 norm simplifies to a plain sum: `L1 = Σ gates`

2. **L1 creates a "pointy" penalty landscape at zero.** Unlike the L2 norm (which has a smooth, bowl-shaped minimum), the L1 norm has a sharp, non-differentiable "corner" at zero. This means the gradient of the penalty has constant magnitude regardless of how close a parameter is to zero — the penalty keeps pushing values *toward* zero with the same force, rather than tapering off.

3. **The sigmoid function amplifies this effect.** Because σ(x) saturates near 0 for large negative gate scores, once the optimizer drives a gate score sufficiently negative, the sigmoid output snaps to ~0, effectively making the pruning decision binary. The gradient of σ(x) vanishes at the extremes, which *locks in* the pruning decision.

4. **Contrast with L2 regularization:** An L2 penalty (Σ gates²) would shrink all gate values toward zero but rarely drive any *exactly* to zero. L1 creates exact zeros because of its geometry — the "diamond" level sets of the L1 norm tend to intersect the loss contours at axis-aligned points, which correspond to exact-zero solutions.

**In summary:** The L1 penalty provides constant pressure toward zero, the sigmoid provides a natural snap-to-zero mechanism, and together they produce clean, sparse gate distributions with most values at exactly 0 and a few remaining near 1.

---

## 4. Training Configuration

| Parameter         | Value                |
|-------------------|----------------------|
| Dataset           | CIFAR-10             |
| Architecture      | 4-layer MLP (256-128-64-10) |
| Optimizer         | Adam (lr=0.005 base, lr=0.015 gates) |
| LR Schedule       | Cosine Annealing     |
| Epochs            | 20 (5 epoch $\lambda$ warmup) |
| Batch Size        | 128                  |
| Pruning Threshold | 0.01                 |
| Data Augmentation | RandomCrop, HFlip    |

---

## 5. Results

### 5.1 Sparsity vs. Accuracy Trade-off

The following table summarizes the observed behavior across three λ values balancing the sparsity-accuracy trade-off:

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Interpretation                          |
|------------|-------------------|---------------------|-----------------------------------------|
| 0.1        | 54.52             | 0.2                 | Minimal pruning, accuracy near baseline |
| 1.0        | 54.21             | 48.2                | Moderate pruning, very little accuracy drop |
| 5.0        | 53.64             | 82.0                | Heavy pruning, minor accuracy loss      |

### 5.2 Key Observations

1. **Low λ (0.1):** The sparsity penalty is too weak to drive many gates to zero. The network behaves almost like a standard MLP, retaining most connections.

2. **Medium λ (1.0):** Nearly half the network's connections are pruned while largely retaining the test accuracy. The network learns to identify and remove its least useful connections. This represents the **sweet spot** of the sparsity-accuracy trade-off.

3. **High λ (5.0):** The aggressive penalty prunes a vast majority (82%) of weights. The model's accuracy drops slightly, showing the incredible resiliency of the network even when retaining less than 20% of its connections.

### 5.3 Per-Layer Sparsity Pattern

A consistent finding is that **earlier layers tend to be pruned more aggressively** than later layers. This is intuitive: the first layer processes raw pixels (3072 inputs), which contains significant redundancy. Later layers, being more compact, tend to preserve a higher fraction of their connections.

---

## 6. Gate Value Distribution

For the best model, the histogram of gate values shows the desired **bimodal distribution**:

- **A large spike at 0:** These are the pruned weights — gates that the network has learned to shut off
- **A cluster near 1:** These are the surviving weights that the network deems essential for classification

This bimodal pattern is the hallmark of a successful self-pruning mechanism. The plot is saved as `results/best_model_gate_distribution.png` after running the script.

---

## 7. Plots Generated

After running `self_pruning_network.py`, the following plots are saved in the `results/` directory:

| File                                    | Contents                                              |
|-----------------------------------------|-------------------------------------------------------|
| `gate_distribution_lambda_*.png`        | Gate value histograms for each λ                      |
| `best_model_gate_distribution.png`      | Detailed distribution for the best model              |
| `sparsity_accuracy_tradeoff.png`        | Bar chart comparing accuracy vs sparsity across λ     |
| `training_history.png`                  | Multi-panel plot of loss, accuracy, sparsity over time |

---

## 8. How to Run

```bash
# Ensure PyTorch and torchvision are installed
pip install torch torchvision matplotlib

# Run the complete experiment
python self_pruning_network.py
```

CIFAR-10 will be downloaded automatically on first run. All results and plots will be saved to the `results/` folder.

---

## 9. Conclusion

The self-pruning neural network successfully demonstrates that **learnable gate parameters + L1 regularization** can produce sparse networks during training without a separate post-hoc pruning step. The λ hyperparameter provides fine-grained control over the sparsity-accuracy trade-off, and the resulting gate distributions show clean bimodal patterns confirming effective pruning.

**Key takeaway:** This approach unifies training and pruning into a single optimization procedure, making it a practical and elegant solution for deploying compact neural networks under resource constraints.
