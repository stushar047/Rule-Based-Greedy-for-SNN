# sgd_snn_baseline.py
# Single-layer SNN baseline trained with surrogate gradients (PyTorch)
# Author: you :)

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------
# Data utilities
# ---------------------------
def load_dataset(name: str):
    name = name.lower()
    if name == "iris":
        D = load_iris()
    elif name == "wine":
        D = load_wine()
    elif name in ("bc", "breast_cancer", "breast-cancer"):
        D = load_breast_cancer()
    else:
        raise ValueError("dataset must be: iris | wine | bc")
    X = D.data.astype(np.float32)
    y = D.target.astype(np.int64)
    classes = len(np.unique(y))
    features = X.shape[1]
    X = MinMaxScaler().fit_transform(X).astype(np.float32)
    return X, y, features, classes

def rate_encode(X01: np.ndarray, T: int) -> np.ndarray:
    """
    Deterministic rate code in [0,1]:
    produce approximately v*T spikes over T steps, spread evenly.
    Returns [N, T, F] binary spikes (time-major for convenience in PyTorch loop).
    """
    N, F = X01.shape
    spikes = np.zeros((N, T, F), dtype=np.float32)
    for n in range(N):
        for f in range(F):
            v = float(X01[n, f])
            k = int(np.round(v * T))
            if k <= 0:
                continue
            idx = np.linspace(0, T-1, num=k, dtype=int)
            spikes[n, idx, f] = 1.0
    return spikes

# ---------------------------
# Surrogate spike function
# ---------------------------
class SpikeFnSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        # fast-sigmoid derivative approximation (triangle-like)
        grad = grad_output * (alpha / (1.0 + (alpha * x).abs()).pow(2))
        return grad, None

def spike_fn(x, alpha=5.0):
    return SpikeFnSTE.apply(x, alpha)

# ---------------------------
# Single-layer SNN module
# ---------------------------
class SingleLayerSNN(nn.Module):
    """
    Single output layer SNN:
      - Weighted input current from spikes
      - Membrane V
      - Thresholding -> spike
      - Absolute refractory (per-class, constant)
      - Reset on spike
    Trains weights W via surrogate-grad through spike nonlinearity.
    """
    def __init__(self, in_features, out_classes,
                 v_th=0.4, v_reset=0.0, refrac=3, alpha=5.0):
        super().__init__()
        self.in_features = in_features
        self.out_classes = out_classes
        self.W = nn.Parameter(torch.empty(in_features, out_classes))
        nn.init.kaiming_normal_(self.W)  # He init
        self.v_th = v_th
        self.v_reset = v_reset
        self.refrac = refrac
        self.alpha = alpha

    def forward(self, x_seq):
        """
        x_seq: [B, T, F] binary spikes
        Returns:
          spike_counts: [B, C]
          spike_traces (optional): None (for simplicity)
        """
        B, T, F = x_seq.shape
        W = self.W  # [F,C]
        V = torch.zeros(B, self.out_classes, device=x_seq.device)
        refr = torch.zeros(B, self.out_classes, device=x_seq.device)  # refractory counters
        spike_counts = torch.zeros(B, self.out_classes, device=x_seq.device)

        for t in range(T):
            x_t = x_seq[:, t, :]  # [B,F]
            I_t = x_t @ W         # [B,C]

            # neurons in refractory ignore input
            active = (refr <= 0).float()
            V = V + I_t * active

            # Surrogate spike
            s_in = V - self.v_th
            s = spike_fn(s_in, self.alpha) * active  # [B,C]

            # Reset and set refractory
            fired = (s > 0)
            if fired.any():
                V = torch.where(fired, torch.full_like(V, self.v_reset), V)
                # increase spike counts
                spike_counts = spike_counts + s
                # set refractory
                refr = torch.where(fired, torch.full_like(refr, float(self.refrac)), refr)

            # decrement refractory
            refr = torch.clamp(refr - 1.0, min=0.0)

        return spike_counts  # [B,C]

# ---------------------------
# Train / Eval
# ---------------------------
def top1_from_counts(spike_counts: torch.Tensor) -> torch.Tensor:
    return spike_counts.argmax(dim=1)

def train_one_epoch(model, loader, optimizer, device, loss_on="counts_ce"):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device)  # [B, T, F]
        yb = yb.to(device)  # [B]
        optimizer.zero_grad()
        counts = model(xb)  # [B, C]

        if loss_on == "counts_ce":
            loss = ce(counts, yb)
        else:
            raise ValueError("Unsupported loss_on")

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = top1_from_counts(counts)
            correct += (pred == yb).sum().item()
            total += yb.numel()
            loss_sum += loss.item() * yb.numel()
    return correct / total, loss_sum / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    all_counts = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        counts = model(xb)
        all_counts.append(counts.cpu())
        pred = top1_from_counts(counts)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    all_counts = torch.cat(all_counts, dim=0)  # [N, C]
    return correct / total, all_counts.numpy()

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="wine",
                    help="iris | wine | bc")
    ap.add_argument("--T", type=int, default=32, help="timesteps")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--refrac", type=int, default=3, help="absolute refractory (timesteps)")
    ap.add_argument("--vth", type=float, default=0.4)
    ap.add_argument("--vre", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=5.0, help="surrogate slope")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    X, y, F, C = load_dataset(args.dataset)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    Xtr_spk = rate_encode(Xtr, args.T)  # [N,T,F]
    Xte_spk = rate_encode(Xte, args.T)

    tr_ds = TensorDataset(torch.from_numpy(Xtr_spk), torch.from_numpy(ytr))
    te_ds = TensorDataset(torch.from_numpy(Xte_spk), torch.from_numpy(yte))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    te_ld = DataLoader(te_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    # Model
    model = SingleLayerSNN(
        in_features=F, out_classes=C,
        v_th=args.vth, v_reset=args.vre,
        refrac=args.refrac, alpha=args.alpha
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_te = -1.0
    for ep in range(1, args.epochs + 1):
        tr_acc, tr_loss = train_one_epoch(model, tr_ld, opt, device, loss_on="counts_ce")
        te_acc, _ = evaluate(model, te_ld, device)
        best_te = max(best_te, te_acc)
        print(f"[{args.dataset.upper()}] Epoch {ep:03d}/{args.epochs} | "
              f"Train Acc {tr_acc:.3f} Loss {tr_loss:.3f} | Test Acc {te_acc:.3f} | Best {best_te:.3f}")

    # Final eval + spike stats
    tr_acc, tr_counts = evaluate(model, tr_ld, device)
    te_acc, te_counts = evaluate(model, te_ld, device)
    print("\n=== Surrogate-Gradient Single-Layer SNN (Baseline) ===")
    print(f"Dataset: {args.dataset} | T={args.T} | Refrac={args.refrac} | Alpha={args.alpha}")
    print(f"Final Train Acc: {tr_acc:.3f} | Test Acc: {te_acc:.3f}")

    # Spike stats (sum across classes gives total output spikes per sample)
    te_total_spikes = te_counts.sum(axis=1)
    print(f"Mean output spikes/sample (test): {te_total_spikes.mean():.2f} ± {te_total_spikes.std():.2f}")

    # Plot spike count distribution per class neuron
    plt.figure(figsize=(8,4))
    for c in range(C):
        plt.hist(te_counts[:, c], bins=20, alpha=0.5, label=f"class-neuron {c}")
    plt.xlabel("Spike count (per output neuron)")
    plt.ylabel("Samples")
    plt.title(f"{args.dataset.upper()} | SNN spike counts per output neuron (T={args.T}, refrac={args.refrac})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"snn_{args.dataset}_spike_hist_T{args.T}_r{args.refrac}.png", dpi=200)
    print(f"Saved: snn_{args.dataset}_spike_hist_T{args.T}_r{args.refrac}.png")

if __name__ == "__main__":
    main()
