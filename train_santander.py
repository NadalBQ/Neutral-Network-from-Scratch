import numpy as np
import pandas as pd

# tu framework
from nn.layers import Dense
from nn.activation import ReLU, Sigmoid
from nn.losses import BinaryCrossEntropy, WeightedBinaryCrossEntropy
from nn.optim import Adam
from nn.model import MLP


import numpy as np

def roc_auc_np(y_true, y_score) -> float:
    """
    ROC-AUC desde cero (sin sklearn).

    y_true: (N,1) o (N,) con 0/1
    y_score: (N,1) o (N,) scores/probabilidades (más alto = más probable clase 1)

    Devuelve AUC en [0,1]. Si no hay positivos o no hay negativos, devuelve nan.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Ranking con manejo de empates: AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    order = np.argsort(y_score)
    scores_sorted = y_score[order]
    y_sorted = y_true[order]

    ranks = np.empty_like(scores_sorted, dtype=np.float64)
    i = 0
    r = 1.0
    N = len(scores_sorted)

    while i < N:
        j = i
        # grupo de empates
        while j + 1 < N and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        # rank medio para empates: (r + (r + (j-i))) / 2
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[i:j + 1] = avg_rank
        r += (j - i + 1)
        i = j + 1

    sum_ranks_pos = float(np.sum(ranks[y_sorted == 1]))
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)
    return float(auc)


def train_val_test_split(X, y, seed=0, p_train=0.8, p_val=0.1):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(p_train * n)
    n_val = int(p_val * n)

    i_tr = idx[:n_train]
    i_va = idx[n_train:n_train + n_val]
    i_te = idx[n_train + n_val:]

    return X[i_tr], y[i_tr], X[i_va], y[i_va], X[i_te], y[i_te]


def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return mu, std


def standardize_apply(X, mu, std):
    return (X - mu) / std


def make_batches(X, y, batch_size=512, shuffle=True, seed=0):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, N, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def compute_metrics(y_hat, y, thr=0.5):
    pred = (y_hat >= thr).astype(int)

    tp = np.sum((pred == 1) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))

    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    f1 = 2 * precision * recall / (precision + recall + 1e-15)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-15)

    return float(acc), float(precision), float(recall), float(f1)


def train_epoch(model, loss_fn, optim, X, y, batch_size=512):
    losses = []
    for Xb, yb in make_batches(X, y, batch_size=batch_size):
        y_hat = model.forward(Xb)
        loss = loss_fn.forward(y_hat, yb)

        dA = loss_fn.backward()
        model.backward(dA)
        optim.step(model.capas)

        losses.append(loss)

    return float(np.mean(losses))


def eval_epoch(model, loss_fn, X, y):
    y_hat = model.forward(X)
    loss = loss_fn.forward(y_hat, y)
    return loss, y_hat


# ===================== MAIN =====================

if __name__ == "__main__":

    df = pd.read_csv("train.csv")
    df.columns = df.columns.str.strip()

    y = df["TARGET"].to_numpy(dtype=np.float64).reshape(-1, 1)
    X = df.drop(columns=["TARGET"]).to_numpy(dtype=np.float64)

    Xtr, ytr, Xva, yva, Xte, yte = train_val_test_split(X, y, seed=42)

    mu, std = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mu, std)
    Xva = standardize_apply(Xva, mu, std)
    Xte = standardize_apply(Xte, mu, std)

    # ======== CLASS WEIGHTS ========
    N = ytr.shape[0]
    N1 = np.sum(ytr == 1)
    N0 = np.sum(ytr == 0)

    w1 = 6.0
    w0 = 1.0

    print("w0:", w0, "w1:", w1)

    # ======== MODEL ========
    d_in = Xtr.shape[1]
    model = MLP([
        Dense(l2=1e-4, d_in=d_in, d_out=256, activation="relu"), ReLU(),
        Dense(l2=1e-4, d_in=256, d_out=128, activation="relu"), ReLU(),
        Dense(l2=1e-4, d_in=128, d_out=1, activation="sigmoid"), Sigmoid(),
    ])

    loss_fn = WeightedBinaryCrossEntropy(w0=w0, w1=w1)
    optim = Adam(lr=1e-3)

    epochs = 10

    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, loss_fn, optim, Xtr, ytr)
        va_loss, y_hat_va = eval_epoch(model, loss_fn, Xva, yva)

        acc, p, r, f1 = compute_metrics(y_hat_va, yva, thr=0.5)

        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} | "
              f"val loss {va_loss:.4f} | acc {acc:.3f} P {p:.3f} R {r:.3f} F1 {f1:.3f}")

    # ======== BEST THRESHOLD SEARCH ========
    print("\nSearching best threshold...")
    best_thr = 0.5
    best_f1 = -1

    _, y_hat_va = eval_epoch(model, loss_fn, Xva, yva)

    for thr in np.linspace(0.05, 0.5, 10):
        acc, p, r, f1 = compute_metrics(y_hat_va, yva, thr=thr)
        print(f"thr={thr:.2f} | P={p:.3f} R={r:.3f} F1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print("\nBest threshold:", best_thr)

    # ======== TEST ========
    te_loss, y_hat_te = eval_epoch(model, loss_fn, Xte, yte)
    auc = roc_auc_np(yte, y_hat_te)
    print("ROC-AUC:", auc)

    acc, p, r, f1 = compute_metrics(y_hat_te, yte, thr=best_thr)

    print(f"\nTEST | loss {te_loss:.4f} | acc {acc:.3f} "
          f"P {p:.3f} R {r:.3f} F1 {f1:.3f}")