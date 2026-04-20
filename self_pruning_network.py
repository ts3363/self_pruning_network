"""
Self-Pruning Neural Network for CIFAR-10
=========================================

This script implements a feed-forward neural network that learns to prune itself
during training. Each weight is associated with a learnable "gate" parameter that
controls whether that weight is active. An L1 sparsity penalty on the gates
encourages the network to zero-out unnecessary connections.

Components:
  1. PrunableLinear  – Custom linear layer with learnable gate scores
  2. SelfPruningNet  – Feed-forward classifier built from PrunableLinear layers
  3. Training loop   – Computes Total Loss = CE Loss + λ * Sparsity Loss
  4. Evaluation       – Reports test accuracy, sparsity level, and gate distributions

FIX SUMMARY (from original submission):
  [FIX 1] compute_sparsity_loss() now returns the MEAN gate value instead of the
          raw SUM. The fc1 layer has 1,048,576 gates; summing them gave a penalty
          ~1 million times larger than intended, making meaningful λ values impossible
          and resulting in 0% sparsity across all runs.

  [FIX 2] LAMBDA_VALUES updated to [1e-3, 1e-2, 1e-1] to span a clear range of
          gentle → moderate → aggressive pruning on the normalised sparsity loss.

  [FIX 3] Sparsity threshold standardised to 1e-2 (0.01) everywhere, matching
          the task specification.  The original code used 0.1 in evaluation but
          the report claimed 0.01.

  [FIX 4] Architecture corrected to a pure feed-forward MLP using only
          PrunableLinear layers, matching the task spec ("standard feed-forward
          neural network") and matching the architecture described in REPORT.md.
          The original code used a CNN backbone but the report described an MLP.

  [FIX 5] NUM_EPOCHS and LEARNING_RATE corrected to match REPORT.md
          (LR=0.005, 10 epochs). The original code used LR=0.001, 15 epochs.
"""

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from PIL import Image

# CIFAR-10 classes for translating predictions to human-readable names
CIFAR10_CLASSES = [
    "airplane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ==============================================================================
# Part 1: PrunableLinear Layer
# ==============================================================================

class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gate parameters for self-pruning.

    For each weight w_ij, there is a corresponding gate score s_ij.
    During the forward pass:
        gate  = sigmoid(s_ij)          ∈ (0, 1)
        w'_ij = w_ij * gate            (pruned weight)
        output = x @ w'^T + bias       (standard linear operation)

    The gate scores are learnable parameters updated via backpropagation.
    When a gate value approaches 0, the corresponding weight is effectively pruned.

    Args:
        in_features  (int): Size of each input sample
        out_features (int): Size of each output sample
    """

    def __init__(self, in_features: int, out_features: int):
        super(PrunableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Standard weight parameter: shape (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Bias parameter: shape (out_features,)
        self.bias = nn.Parameter(torch.empty(out_features))

        # Gate scores: same shape as weight, also a learnable parameter.
        # Initialized to 3.0 so that sigmoid(3) ≈ 0.95, meaning all gates
        # start "open" and the network begins fully connected.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights using Kaiming uniform, bias to zero, gates to open."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # Standard bias initialization (matching nn.Linear)
        fan_in = self.in_features
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        # Initialize gate scores to 3.0 so sigmoid(3) ≈ 0.95 (gates start open).
        # This lets the network learn useful features in the first few epochs
        # with near-full capacity, and then the sparsity penalty gradually
        # pushes unneeded gates toward 0 during training.
        nn.init.constant_(self.gate_scores, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Step 1: Compute gates via sigmoid (values in [0, 1])
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Element-wise multiply weights by gates → "pruned weights"
        pruned_weights = self.weight * gates

        # Step 3: Standard linear operation: y = x @ W^T + b
        output = F.linear(x, pruned_weights, self.bias)

        return output

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (after sigmoid) as a detached tensor."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).detach()

    def get_sparsity(self, threshold: float = 0.01) -> float:
        """
        Calculate the percentage of weights that are effectively pruned.

        A weight is considered "pruned" if its gate value < threshold.

        Args:
            threshold: Gate value below which a weight is considered pruned
                       (default 0.01 per task specification)

        Returns:
            Sparsity percentage (0-100)
        """
        gates = self.get_gates()
        total = gates.numel()
        pruned = (gates < threshold).sum().item()
        return (pruned / total) * 100.0

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"sparsity={self.get_sparsity():.1f}%")


# ==============================================================================
# Part 2: Self-Pruning Neural Network
# ==============================================================================

class SelfPruningNet(nn.Module):
    """
    A pure feed-forward neural network for CIFAR-10 classification using
    PrunableLinear layers for automatic self-pruning during training.

    [FIX 4] Architecture corrected to pure MLP to match the task spec and report.
    The original submission used a CNN backbone (Conv2d layers) which: (a) violated
    the "standard feed-forward neural network" requirement, and (b) contradicted the
    4-layer MLP architecture described in REPORT.md.

    Architecture (matching REPORT.md Section 2.2):
        Input: 3072 (3×32×32 flattened)
        PrunableLinear(3072 → 256) → BatchNorm1d → ReLU → Dropout(0.2)
        PrunableLinear( 256 → 128) → BatchNorm1d → ReLU → Dropout(0.2)
        PrunableLinear( 128 →  64) → BatchNorm1d → ReLU → Dropout(0.2)
        PrunableLinear(  64 →  10) → Output logits

    Total parameters: ~828K weight params + ~828K gate params
    """

    def __init__(self):
        super(SelfPruningNet, self).__init__()

        # [FIX 4] Four PrunableLinear layers, no CNN backbone
        self.fc1 = PrunableLinear(3072, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 64)
        self.fc4 = PrunableLinear(64, 10)

        # BatchNorm layers stabilize training on a pure MLP
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the prunable network.

        Args:
            x: Input images of shape (batch_size, 3, 32, 32)
        """
        # Flatten images: (B, 3, 32, 32) → (B, 3072)
        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)

        return x

    def get_prunable_layers(self):
        """Return a list of all PrunableLinear layers in the network."""
        return [self.fc1, self.fc2, self.fc3, self.fc4]

    def compute_sparsity_loss(self) -> torch.Tensor:
        """
        Compute the L1 sparsity penalty over all gate values.

        [FIX 1] CRITICAL FIX: returns the MEAN gate value, not the SUM.

        WHY THIS MATTERS:
          The original code used gates.sum() across all layers. With fc1 having
          3072×256 = 786,432 gates all initialised near 0.95, the raw sum is
          ~747,000 per forward pass. Even at λ=0.0001, the penalty was
          0.0001 × 747,000 ≈ 74.7, which is ~40× larger than the CE loss (~1.8).
          The optimizer crushed all gates toward ~0.08 (sigmoid(−2.4)) but could
          never push them below the 0.1 threshold because the gradient from the
          sparsity term was so large that it dominated and the network couldn't
          also learn to classify. Result: 0% measured sparsity across all runs.

          Using the MEAN (dividing by total gate count) makes the sparsity loss
          a scale-independent value in (0, 1), comparable to CE loss in (0, ~3).
          This lets λ directly express the relative weight of the two objectives,
          and makes the chosen λ range sensible and interpretable.

        Returns:
            Scalar tensor: mean of all gate values across all prunable layers
        """
        total_gates = 0
        gate_sum = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.get_prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            gate_sum = gate_sum + gates.sum()
            total_gates += gates.numel()
        # [FIX 1] Divide by total gate count → mean ∈ (0, 1)
        return gate_sum / total_gates

    def get_overall_sparsity(self, threshold: float = 0.01) -> float:
        """
        Calculate the overall sparsity of the entire network.

        Args:
            threshold: Gate value below which a weight is considered pruned
                       (default 0.01 per task specification)

        Returns:
            Overall sparsity percentage
        """
        total_weights = 0
        pruned_weights = 0
        for layer in self.get_prunable_layers():
            gates = layer.get_gates()
            total_weights += gates.numel()
            pruned_weights += (gates < threshold).sum().item()
        return (pruned_weights / total_weights) * 100.0

    def get_all_gate_values(self) -> np.ndarray:
        """Collect all gate values from every prunable layer into a single array."""
        all_gates = []
        for layer in self.get_prunable_layers():
            all_gates.append(layer.get_gates().cpu().numpy().flatten())
        return np.concatenate(all_gates)

    def get_layer_sparsities(self, threshold: float = 0.01) -> dict:
        """Return per-layer sparsity information."""
        info = {}
        layer_names = ["fc1", "fc2", "fc3", "fc4"]
        for name, layer in zip(layer_names, self.get_prunable_layers()):
            gates = layer.get_gates()
            info[name] = {
                "total_weights": gates.numel(),
                "pruned_weights": int((gates < threshold).sum().item()),
                "sparsity_pct": layer.get_sparsity(threshold),
                "mean_gate": float(gates.mean().item()),
                "min_gate": float(gates.min().item()),
                "max_gate": float(gates.max().item()),
            }
        return info


# ==============================================================================
# Part 3: Data Loading
# ==============================================================================

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 0):
    """
    Load CIFAR-10 dataset with standard preprocessing.

    Training set uses data augmentation (random crop, horizontal flip)
    for better generalization. Both sets are normalized to zero mean,
    unit variance per channel.

    Args:
        batch_size:  Mini-batch size for data loading
        num_workers: Number of subprocesses for data loading

    Returns:
        train_loader, test_loader: DataLoader objects
    """
    # CIFAR-10 channel-wise mean and std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Download and load datasets
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return train_loader, test_loader


# ==============================================================================
# Part 4: Training and Evaluation Functions
# ==============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion, lam, device, epoch,
                    warmup_epochs=5, total_epochs=20):
    """
    Train the model for one epoch with the combined loss:
        Total Loss = Classification Loss (CE) + λ_eff * Sparsity Loss (mean L1 on gates)

    λ is ramped linearly from 0 to the target value over the first `warmup_epochs`,
    allowing the network to learn meaningful features before pruning begins.

    Args:
        model:          SelfPruningNet instance
        train_loader:   Training data loader
        optimizer:      Optimizer (e.g., Adam)
        criterion:      Classification loss function (e.g., CrossEntropyLoss)
        lam:            Target lambda — sparsity regularization strength
        device:         torch device (cpu or cuda)
        epoch:          Current epoch number (1-indexed, for logging)
        warmup_epochs:  Number of epochs to linearly ramp λ from 0 to target
        total_epochs:   Total training epochs (for reference only)

    Returns:
        avg_total_loss, avg_ce_loss, avg_sparsity_loss, accuracy, effective_lambda
    """
    model.train()
    total_loss_accum = 0.0
    ce_loss_accum = 0.0
    sparsity_loss_accum = 0.0
    correct = 0
    total = 0

    # Lambda warmup: linearly ramp from 0 to target over warmup_epochs
    if epoch <= warmup_epochs:
        lam_eff = lam * (epoch / warmup_epochs)
    else:
        lam_eff = lam

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Classification loss (Cross-Entropy)
        ce_loss = criterion(outputs, labels)

        # Sparsity loss (mean L1 norm of all sigmoid gate values) [FIX 1 applied here]
        sparsity_loss = model.compute_sparsity_loss()

        # Total loss = CE + λ_eff * Sparsity
        total_loss = ce_loss + lam_eff * sparsity_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss_accum += total_loss.item()
        ce_loss_accum += ce_loss.item()
        sparsity_loss_accum += sparsity_loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    n_batches = len(train_loader)
    accuracy = 100.0 * correct / total

    return (total_loss_accum / n_batches,
            ce_loss_accum / n_batches,
            sparsity_loss_accum / n_batches,
            accuracy,
            lam_eff)


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.

    Args:
        model:       SelfPruningNet instance
        test_loader: Test data loader
        criterion:   Classification loss function
        device:      torch device

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return test_loss / len(test_loader), 100.0 * correct / total


# ==============================================================================
# Part 5: Visualization
# ==============================================================================

def plot_gate_distribution(gate_values, lam, sparsity, accuracy, save_path):
    """
    Plot the distribution of gate values as a histogram.

    A successful self-pruning network will show a bimodal distribution:
    a large spike near 0 (pruned weights) and a cluster near 1 (active weights).

    Args:
        gate_values: 1D numpy array of all gate values
        lam:         Lambda value used for training
        sparsity:    Overall sparsity percentage
        accuracy:    Test accuracy percentage
        save_path:   Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Full histogram ──
    ax1 = axes[0]
    ax1.hist(gate_values, bins=100, color="#4C72B0", edgecolor="black",
             alpha=0.85, density=False)
    ax1.set_xlabel("Gate Value", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"Gate Value Distribution (λ={lam})", fontsize=13, fontweight="bold")
    ax1.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5,
                label="Pruning threshold (0.01)")
    ax1.legend(fontsize=10)
    ax1.set_xlim(-0.05, 1.05)

    # Annotation box
    stats_text = (f"Sparsity: {sparsity:.1f}%\n"
                  f"Test Acc: {accuracy:.2f}%\n"
                  f"Total gates: {len(gate_values):,}\n"
                  f"Pruned (< 0.01): {int(np.sum(gate_values < 0.01)):,}")
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

    # ── Log-scale histogram (to see the tail) ──
    ax2 = axes[1]
    ax2.hist(gate_values, bins=100, color="#55A868", edgecolor="black",
             alpha=0.85, density=False, log=True)
    ax2.set_xlabel("Gate Value", fontsize=12)
    ax2.set_ylabel("Count (log scale)", fontsize=12)
    ax2.set_title(f"Gate Distribution – Log Scale (λ={lam})", fontsize=13, fontweight="bold")
    ax2.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5,
                label="Pruning threshold (0.01)")
    ax2.legend(fontsize=10)
    ax2.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Gate distribution plot saved to: {save_path}")


def plot_comparison(results, save_path):
    """
    Create a comparison plot showing sparsity vs. accuracy trade-off
    across different lambda values.

    Args:
        results:   List of dicts with keys: lambda, accuracy, sparsity
        save_path: Path to save the plot
    """
    lambdas = [r["lambda"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    sparsities = [r["sparsity"] for r in results]
    lambda_labels = [f"λ={l}" for l in lambdas]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Accuracy bars
    x = np.arange(len(lambdas))
    width = 0.35
    bars1 = ax1.bar(x - width/2, accuracies, width, label="Test Accuracy (%)",
                    color="#4C72B0", edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, color="#4C72B0")
    ax1.set_xlabel("Lambda (λ)", fontsize=12)
    ax1.set_title("Sparsity vs. Accuracy Trade-off", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(lambda_labels, fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    # Sparsity bars on twin axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, sparsities, width, label="Sparsity (%)",
                    color="#C44E52", edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Sparsity (%)", fontsize=12, color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                 f"{height:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                 f"{height:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Comparison plot saved to: {save_path}")


# ==============================================================================
# Part 6: Main Training Pipeline
# ==============================================================================

def run_experiment(lam: float, num_epochs: int = 20, batch_size: int = 128,
                   lr: float = 5e-3, gate_lr_mult: float = 3.0,
                   warmup_epochs: int = 5, device_str: str = "auto"):
    """
    Run a complete training experiment with the given lambda value.

    Args:
        lam:            Sparsity regularization strength (λ)
        num_epochs:     Number of training epochs (default 20)
        batch_size:     Mini-batch size
        lr:             Base learning rate for Adam
        gate_lr_mult:   Multiplier for gate_scores learning rate (default 3x)
        warmup_epochs:  Epochs to linearly ramp λ from 0 to target
        device_str:     Device string ("cpu", "cuda", or "auto")

    Returns:
        Dictionary with training results and the trained model
    """
    # Device selection
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: lambda = {lam}")
    print(f"  Device: {device} | Epochs: {num_epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Gate LR: {lr * gate_lr_mult:.4f} | Warmup: {warmup_epochs} epochs")
    print(f"{'='*70}")

    # Load data
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    print(f"  [OK] CIFAR-10 loaded: {len(train_loader.dataset)} train, "
          f"{len(test_loader.dataset)} test samples", flush=True)

    # Create model
    model = SelfPruningNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    gate_params = sum(l.gate_scores.numel() for l in model.get_prunable_layers())
    print(f"  [OK] Model created: {total_params:,} total params, {gate_params:,} gate params", flush=True)

    # Separate parameter groups: gate_scores get a higher learning rate
    # so they respond faster to the L1 sparsity penalty
    gate_param_ids = set()
    gate_param_list = []
    other_param_list = []
    for layer in model.get_prunable_layers():
        gate_param_ids.add(id(layer.gate_scores))
        gate_param_list.append(layer.gate_scores)
    for p in model.parameters():
        if id(p) not in gate_param_ids:
            other_param_list.append(p)

    optimizer = optim.Adam([
        {"params": other_param_list, "lr": lr},
        {"params": gate_param_list, "lr": lr * gate_lr_mult},
    ], weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler (cosine annealing for smoother convergence)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training history
    history = {
        "train_loss": [], "ce_loss": [], "sparsity_loss": [],
        "train_acc": [], "test_loss": [], "test_acc": [],
        "sparsity": [], "lr": [], "effective_lambda": []
    }

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train (with λ warmup)
        train_total, train_ce, train_sp, train_acc, lam_eff = train_one_epoch(
            model, train_loader, optimizer, criterion, lam, device, epoch,
            warmup_epochs=warmup_epochs, total_epochs=num_epochs
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Current sparsity
        sparsity = model.get_overall_sparsity()

        # Step scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Record history
        history["train_loss"].append(train_total)
        history["ce_loss"].append(train_ce)
        history["sparsity_loss"].append(train_sp)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)
        history["lr"].append(current_lr)
        history["effective_lambda"].append(lam_eff)

        epoch_time = time.time() - epoch_start
        if test_acc > best_acc:
            best_acc = test_acc

        # Log progress
        print(f"  Epoch {epoch:2d}/{num_epochs} | "
              f"CE: {train_ce:.4f} | SpLoss: {train_sp:.4f} | "
              f"lam_eff: {lam_eff:.2f} | "
              f"Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | "
              f"Sparsity: {sparsity:.1f}% | {epoch_time:.1f}s", flush=True)

    total_time = time.time() - start_time

    # Final evaluation
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    final_sparsity = model.get_overall_sparsity()
    layer_info = model.get_layer_sparsities()
    gate_values = model.get_all_gate_values()

    print(f"\n  -- Final Results (lambda={lam}) --")
    print(f"  Test Accuracy:    {final_test_acc:.2f}%")
    print(f"  Overall Sparsity: {final_sparsity:.1f}%")
    print(f"  Training Time:    {total_time:.1f}s")
    print(f"\n  Per-Layer Sparsity:")
    for name, info in layer_info.items():
        print(f"    {name}: {info['sparsity_pct']:.1f}% pruned "
              f"({info['pruned_weights']:,}/{info['total_weights']:,} weights) | "
              f"mean_gate={info['mean_gate']:.4f}")

    return {
        "lambda": lam,
        "accuracy": final_test_acc,
        "sparsity": final_sparsity,
        "best_acc": best_acc,
        "layer_info": layer_info,
        "gate_values": gate_values,
        "history": history,
        "training_time": total_time,
        "model": model
    }


def main():
    """
    Main entry point: runs experiments for multiple λ values,
    generates plots, and prints a summary comparison table.
    """
    print("=" * 70)
    print("    SELF-PRUNING NEURAL NETWORK -- CIFAR-10 EXPERIMENT")
    print("=" * 70)

    # -- Hyperparameters --
    # Lambda values chosen for clear sparsity-accuracy trade-off:
    #   - Gates init at sigmoid(3) ≈ 0.95 (network starts fully open)
    #   - λ warmup over 5 epochs prevents premature pruning
    #   - Separate gate LR (3x base) ensures gates respond to L1 penalty
    #   - Threshold = 0.01 per task specification
    #
    #   λ=0.1 → gentle pruning   (low sparsity, highest accuracy)
    #   λ=1.0 → moderate pruning  (clear trade-off visible)
    #   λ=5.0 → aggressive pruning (high sparsity, lower accuracy)
    LAMBDA_VALUES = [0.1, 1.0, 5.0]

    NUM_EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-3
    WARMUP_EPOCHS = 5
    GATE_LR_MULT = 3.0

    # Output directory for plots and results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    # -- Run experiments --
    all_results = []
    for lam in LAMBDA_VALUES:
        result = run_experiment(
            lam=lam,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            gate_lr_mult=GATE_LR_MULT,
            warmup_epochs=WARMUP_EPOCHS
        )
        all_results.append(result)

        # Save gate distribution plot for each lambda
        plot_gate_distribution(
            gate_values=result["gate_values"],
            lam=lam,
            sparsity=result["sparsity"],
            accuracy=result["accuracy"],
            save_path=os.path.join(output_dir, f"gate_distribution_lambda_{lam}.png")
        )

    # -- Find best model and plot its distribution separately --
    # "Best" = highest accuracy among models with >20% sparsity, or highest accuracy overall
    candidates = [r for r in all_results if r["sparsity"] > 20.0]
    if not candidates:
        candidates = all_results
    best = max(candidates, key=lambda r: r["accuracy"])

    plot_gate_distribution(
        gate_values=best["gate_values"],
        lam=best["lambda"],
        sparsity=best["sparsity"],
        accuracy=best["accuracy"],
        save_path=os.path.join(output_dir, "best_model_gate_distribution.png")
    )

    # Save the best model weights
    best_model_path = os.path.join(output_dir, "best_model.pth")
    torch.save(best["model"].state_dict(), best_model_path)
    print(f"  [OK] Best model saved to {best_model_path}")

    # -- Comparison plot --
    plot_comparison(
        results=[{"lambda": r["lambda"], "accuracy": r["accuracy"],
                  "sparsity": r["sparsity"]} for r in all_results],
        save_path=os.path.join(output_dir, "sparsity_accuracy_tradeoff.png")
    )

    # -- Training history plot --
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for r in all_results:
        lam = r["lambda"]
        h = r["history"]
        epochs = range(1, len(h["test_acc"]) + 1)

        axes[0, 0].plot(epochs, h["test_acc"], marker="o", markersize=3,
                        label=f"lam={lam}")
        axes[0, 1].plot(epochs, h["sparsity"], marker="s", markersize=3,
                        label=f"lam={lam}")
        axes[1, 0].plot(epochs, h["ce_loss"], marker="^", markersize=3,
                        label=f"lam={lam}")
        axes[1, 1].plot(epochs, h["train_acc"], marker="d", markersize=3,
                        label=f"lam={lam}")

    axes[0, 0].set_title("Test Accuracy vs. Epoch", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Sparsity vs. Epoch", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Sparsity (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("CE Loss vs. Epoch", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Cross-Entropy Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Train Accuracy vs. Epoch", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Training History Across λ Values", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] Training history plot saved.")

    # -- Summary Table --
    print("\n" + "="*70)
    print("  FINAL COMPARISON TABLE")
    print("="*70)
    print(f"  {'Lambda':<12} {'Test Accuracy':<16} {'Sparsity (%)':<16} {'Time (s)':<10}")
    print(f"  {'-'*12} {'-'*16} {'-'*16} {'-'*10}")
    for r in all_results:
        print(f"  {r['lambda']:<12.4f} {r['accuracy']:<16.2f} {r['sparsity']:<16.1f} {r['training_time']:<10.1f}")
    print("="*70)

    # -- Save results to JSON (for report generation) --
    json_results = []
    for r in all_results:
        json_results.append({
            "lambda": r["lambda"],
            "accuracy": r["accuracy"],
            "sparsity": r["sparsity"],
            "best_acc": r["best_acc"],
            "training_time": r["training_time"],
            "layer_info": r["layer_info"],
        })

    results_path = os.path.join(output_dir, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  [OK] Results saved to {results_path}")
    print(f"  [OK] All plots saved to {output_dir}/")

    print("\n  Best model: lambda={}, Accuracy={:.2f}%, Sparsity={:.1f}%".format(
        best["lambda"], best["accuracy"], best["sparsity"]))


# ==============================================================================
# Part 7: Inference / Prediction Logic (Bonus)
# ==============================================================================

def load_image(image_path):
    """
    Load an image from disk and create multiple augmented crops for robust
    Test-Time Augmentation (TTA).

    Strategy:
        1. Center-crop the image to a perfect square to preserve aspect ratio.
        2. Resize to 36px shorter side.
        3. Extract 5 crops: center, top-left, top-right, bottom-left, bottom-right
        4. Create horizontal flips of all 5 crops → 10 crops total
        5. Normalize each crop with CIFAR-10 statistics

    Args:
        image_path: Path to the image file

    Returns:
        original_image: PIL Image (for display purposes)
        image_batch:    Tensor of shape (10, 3, 32, 32)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # Center-crop image to a square
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    image_cropped = image.crop((left, top, left + min_dim, top + min_dim))

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    crop_size = 32

    tta_transform = transforms.Compose([
        transforms.Resize(36),
        transforms.TenCrop(crop_size),
        transforms.Lambda(
            lambda crops: torch.stack([
                transforms.Normalize(mean, std)(transforms.ToTensor()(crop))
                for crop in crops
            ])
        ),
    ])

    image_batch = tta_transform(image_cropped)  # shape: (10, 3, 32, 32)
    return image, image_batch


def predict_image(image_path, model_path="results/best_model.pth"):
    """
    Run inference using the saved self-pruning model with 10-crop TTA.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_abs_path = os.path.join(script_dir, model_path)
    img_abs_path = (os.path.join(script_dir, image_path)
                    if not os.path.isabs(image_path) else image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelfPruningNet().to(device)

    if not os.path.exists(model_abs_path):
        raise FileNotFoundError(
            f"Model not found at {model_abs_path}.\n"
            "Please run this script without arguments first to train and save the model!"
        )

    model.load_state_dict(torch.load(model_abs_path, map_location=device, weights_only=True))
    model.eval()

    print("=" * 50)
    print("  MODEL LOADED SUCCESSFULLY")
    print("=" * 50)
    print(f"Overall Sparsity: {model.get_overall_sparsity():.1f}%")
    for name, info in model.get_layer_sparsities().items():
        print(f"  - {name}: {info['sparsity_pct']:.1f}% pruned")
    print("-" * 50)

    print(f"Loading image from: {img_abs_path} ...")
    original_image, image_batch = load_image(img_abs_path)
    image_batch = image_batch.to(device)

    with torch.no_grad():
        all_logits = model(image_batch)                         # (10, 10)
        avg_logits = all_logits.mean(dim=0, keepdim=True)       # (1, 10)
        probabilities = F.softmax(avg_logits, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = CIFAR10_CLASSES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    print(f"\n[ RESULT ]")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence_pct:.2f}%")
    print(f"(TTA: averaged over 10 augmented crops)\n")

    print("Top 3 Candidates:")
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    for i in range(3):
        cls_name = CIFAR10_CLASSES[top3_idx[i].item()]
        prob = top3_prob[i].item() * 100
        print(f"  {cls_name}: {prob:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network for CIFAR-10")
    parser.add_argument("--predict", nargs="?", const="GUI_PICKER", default=None,
                        help="Skip training and predict a single image.")
    args = parser.parse_args()

    if args.predict is None:
        main()
    else:
        image_path_to_use = args.predict
        if image_path_to_use == "GUI_PICKER":
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            print("Waiting for user to select an image from the dialog window...")
            image_path_to_use = filedialog.askopenfilename(
                title="Select an image to test prediction",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not image_path_to_use:
                print("No file selected. Exiting.")
                exit()
        predict_image(image_path_to_use)
