# sgd_snn_baseline_constrained_tf_optimized.py
# Single-layer surrogate-gradient SNN (TensorFlow/Keras) with constrained neuron dynamics
# Optimized defaults for training on small datasets

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

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
    y = D.target.astype(np.int32)  # ensure int32 for TF equality ops
    classes = int(len(np.unique(y)))
    features = X.shape[1]
    X = MinMaxScaler().fit_transform(X).astype(np.float32)
    return X, y, features, classes

def rate_encode(X01: np.ndarray, T: int) -> np.ndarray:
    """
    Deterministic rate code in [0,1]: k=round(v*T) spikes spaced across T (linspace).
    Returns [N, T, F] binary spikes.
    """
    N, F = X01.shape
    spikes = np.zeros((N, T, F), dtype=np.float32)
    # small offset to avoid 0 spikes
    X01_scaled = X01 * 0.8 + 0.1
    for n in range(N):
        for f in range(F):
            v = float(X01_scaled[n, f])
            k = int(np.round(v * T))
            if k <= 0:
                continue
            idx = np.linspace(0, T - 1, num=k, dtype=int)
            spikes[n, idx, f] = 1.0
    return spikes

# ---------------------------
# Surrogate spike (fast-sigmoid STE)
# y = 1[x>0] with d/dx ≈ alpha / (1 + |alpha*x|)^2
# ---------------------------
@tf.custom_gradient
def spike_fn(x, alpha):
    y = tf.cast(x > 0.0, x.dtype)
    def grad(dy):
        denom = 1.0 + tf.abs(alpha * x)
        g = alpha / tf.square(denom)
        return dy * g, tf.zeros_like(alpha)
    return y, grad

# ---------------------------
# Single-layer SNN with constraints
# ---------------------------
class SingleLayerSNN(keras.Model):
    def __init__(self,
                 in_features, out_classes,
                 v_th=0.4, v_reset=0.0,
                 leak=0.01, refr_leak=0.01,
                 refr_potential=0.0, std_potential=0.0,
                 refrac=0, alpha=20.0,
                 v_max=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_features = int(in_features)
        self.out_classes = int(out_classes)

        self.W = self.add_weight(
            name="W",
            shape=(self.in_features, self.out_classes),
            initializer=keras.initializers.HeNormal(),
            trainable=True,
            dtype=tf.float32,
        )

        self.v_th = float(v_th)
        self.v_reset = float(v_reset)
        self.leak = float(leak)
        self.refr_leak = float(refr_leak)
        self.refr_pot = float(refr_potential)
        self.std_pot = float(std_potential)
        self.refrac = int(refrac)
        self.alpha = float(alpha)
        self.v_max = None if v_max is None else float(v_max)

    def call(self, x_seq, training=False):
        x_seq = tf.convert_to_tensor(x_seq, dtype=tf.float32)
        B = tf.shape(x_seq)[0]
        T = tf.shape(x_seq)[1]
        F = tf.shape(x_seq)[2]

        V = tf.zeros((B, self.out_classes), dtype=tf.float32)
        refr = tf.zeros((B, self.out_classes), dtype=tf.float32)
        mode = tf.zeros((B, self.out_classes), dtype=tf.int8)
        spike_counts = tf.zeros((B, self.out_classes), dtype=tf.float32)

        v_reset = tf.fill(tf.shape(V), self.v_reset)
        v_th = tf.fill(tf.shape(V), self.v_th)
        std_pot = tf.fill(tf.shape(V), self.std_pot)
        refr_pot = tf.fill(tf.shape(V), self.refr_pot)
        leak = tf.fill(tf.shape(V), self.leak)
        refr_leak = tf.fill(tf.shape(V), self.refr_leak)

        for t in tf.range(T):
            x_t = x_seq[:, t, :]
            I_t = tf.linalg.matmul(x_t, self.W)

            m_std = tf.equal(mode, 0)
            m_abs = tf.equal(mode, 1)
            m_rel = tf.equal(mode, 2)

            # std phase
            if tf.reduce_any(m_std):
                V = tf.where(m_std, V + I_t - leak, V)
                V = tf.where(tf.logical_and(m_std, V < std_pot), std_pot, V)
                if self.v_max is not None:
                    V = tf.minimum(V, self.v_max)

                s_in = V - v_th
                s = spike_fn(s_in, tf.constant(self.alpha, dtype=tf.float32)) * tf.cast(m_std, tf.float32)

                fired = s > 0.0
                if tf.reduce_any(fired):
                    spike_counts = spike_counts + tf.cast(s, tf.float32)
                    V = tf.where(fired, v_reset, V)
                    refr = tf.where(fired, tf.fill(tf.shape(refr), float(self.refrac)), refr)
                    mode = tf.where(fired, tf.ones_like(mode), mode)

            # abs refractory
            if tf.reduce_any(m_abs):
                refr = tf.where(m_abs, tf.maximum(refr - 1.0, 0.0), refr)
                done = tf.logical_and(m_abs, refr <= 0.0)
                mode = tf.where(done, tf.fill(tf.shape(mode), tf.cast(2, tf.int8)), mode)
                V = tf.where(tf.logical_and(m_abs, V < refr_pot), refr_pot, V)

            # rel refractory
            if tf.reduce_any(m_rel):
                V = tf.where(m_rel, V + refr_leak, V)
                V = tf.where(tf.logical_and(m_rel, V < refr_pot), refr_pot, V)
                if self.v_max is not None:
                    V = tf.minimum(V, self.v_max)
                back = tf.logical_and(m_rel, V > std_pot)
                mode = tf.where(back, tf.zeros_like(mode), mode)

        return spike_counts

# ---------------------------
# Train / Eval
# ---------------------------
def top1_from_counts(spike_counts: tf.Tensor) -> tf.Tensor:
    return tf.argmax(spike_counts, axis=1, output_type=tf.int32)

def train_one_epoch(model, dataset, optimizer,
                    spike_reg=0.0, grad_clip=None, wmin=None, wmax=None):
    total, correct, loss_sum = 0, 0, 0.0
    ce = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    for xb, yb in dataset:
        with tf.GradientTape() as tape:
            counts = model(xb, training=True)
            loss = ce(yb, counts)
            if spike_reg > 0.0:
                loss += spike_reg * tf.reduce_mean(counts)

        grads = tape.gradient(loss, model.trainable_variables)
        if grad_clip is not None and grad_clip > 0:
            grads = [tf.clip_by_norm(g, grad_clip) if g is not None else None for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if (wmin is not None) or (wmax is not None):
            newW = tf.clip_by_value(
                model.W,
                wmin if wmin is not None else -np.inf,
                wmax if wmax is not None else np.inf
            )
            model.W.assign(newW)

        preds = top1_from_counts(counts)
        correct += int(tf.reduce_sum(tf.cast(tf.equal(preds, yb), tf.int32)).numpy())
        total += int(yb.shape[0])
        loss_sum += float(loss.numpy()) * int(yb.shape[0])

    return correct / total, loss_sum / total

@tf.function
def _eval_step(model, xb):
    return model(xb, training=False)

def evaluate(model, dataset):
    total, correct = 0, 0
    all_counts = []
    for xb, yb in dataset:
        counts = _eval_step(model, xb)
        preds = top1_from_counts(counts)
        correct += int(tf.reduce_sum(tf.cast(tf.equal(preds, yb), tf.int32)).numpy())
        total += int(yb.shape[0])
        all_counts.append(counts.numpy())
    all_counts = np.concatenate(all_counts, axis=0)
    return correct / total, all_counts

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="wine", help="iris | wine | bc")
    ap.add_argument("--T", type=int, default=64, help="timesteps")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--refrac", type=int, default=0)
    ap.add_argument("--vth", type=float, default=0.4)
    ap.add_argument("--vres", type=float, default=0.0)
    ap.add_argument("--leak", type=float, default=0.01)
    ap.add_argument("--refr_leak", type=float, default=0.01)
    ap.add_argument("--refr_pot", type=float, default=0.0)
    ap.add_argument("--std_pot", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--wmin", type=float, default=-5.0)
    ap.add_argument("--wmax", type=float, default=5.0)
    ap.add_argument("--spike_reg", type=float, default=0.0)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print(f"TensorFlow version: {tf.__version__}")

    # Data
    X, y, F, C = load_dataset(args.dataset)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    Xtr_spk = rate_encode(Xtr, args.T)
    Xte_spk = rate_encode(Xte, args.T)

    tr_ds = tf.data.Dataset.from_tensor_slices((Xtr_spk, ytr)).shuffle(8192, seed=args.seed).batch(args.batch)
    te_ds = tf.data.Dataset.from_tensor_slices((Xte_spk, yte)).batch(args.batch)

    # Model
    model = SingleLayerSNN(
        in_features=F, out_classes=C,
        v_th=args.vth, v_reset=args.vres,
        leak=args.leak, refr_leak=args.refr_leak,
        refr_potential=args.refr_pot, std_potential=args.std_pot,
        refrac=args.refrac, alpha=args.alpha,
        v_max=args.vmax
    )

    opt = keras.optimizers.Adam(learning_rate=args.lr)

    # Train
    best_te = -1.0
    for ep in range(1, args.epochs + 1):
        tr_acc, tr_loss = train_one_epoch(
            model, tr_ds, opt,
            spike_reg=args.spike_reg,
            grad_clip=args.grad_clip,
            wmin=args.wmin, wmax=args.wmax
        )
        te_acc, _ = evaluate(model, te_ds)
        best_te = max(best_te, te_acc)
        print(f"[{args.dataset.upper()}] Ep {ep:03d}/{args.epochs} | "
              f"Train Acc {tr_acc:.3f} Loss {tr_loss:.3f} | Test Acc {te_acc:.3f} | Best {best_te:.3f}")

    # Final eval + spike stats
    tr_acc, tr_counts = evaluate(model, tr_ds)
    te_acc, te_counts = evaluate(model, te_ds)
    print("\n=== Constrained Surrogate-Gradient SNN (Baseline, TF) ===")
    print(f"Dataset: {args.dataset} | T={args.T} | Refrac={args.refrac} | Alpha={args.alpha}")
    print(f"Params: leak={args.leak}, refr_leak={args.refr_leak}, "
          f"refr_pot={args.refr_pot}, std_pot={args.std_pot}, vth={args.vth}, vmax={args.vmax}")
    print(f"Final Train Acc: {tr_acc:.3f} | Test Acc: {te_acc:.3f}")

    # Spike stats
    te_total_spikes = te_counts.sum(axis=1)
    print(f"Mean output spikes/sample (test): {te_total_spikes.mean():.2f} ± {te_total_spikes.std():.2f}")

    # Plot spike count distribution per class neuron
    plt.figure(figsize=(8, 4))
    for c in range(C):
        plt.hist(te_counts[:, c], bins=20, alpha=0.5, label=f"class-neuron {c}")
    plt.xlabel("Spike count (per output neuron)")
    plt.ylabel("Samples")
    plt.title(f"{args.dataset.upper()} | Constrained SNN spike counts (T={args.T}, refrac={args.refrac})")
    plt.legend()
    plt.tight_layout()
    out_png = f"snn_constrained_tf_{args.dataset}_spike_hist_T{args.T}_r{args.refrac}.png"
    plt.savefig(out_png, dpi=220)
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()