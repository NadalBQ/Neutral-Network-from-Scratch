import numpy as np

from sklearn.datasets import fetch_openml

# tu framework
from nn.layers import Dense
from nn.activation import ReLU
from nn.optim import Adam
from nn.model import MLP
from nn.losses import CrossEntropy  # tu clase softmax+CE junta


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


def make_batches(X, y, batch_size=64, shuffle=True, seed=0):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, N, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def one_hot(y_int, C):
    return np.eye(C, dtype=np.float64)[y_int]


def acc_multiclass(logits, y_onehot):
    pred = np.argmax(logits, axis=1)
    true = np.argmax(y_onehot, axis=1)
    return float((pred == true).mean())


def train_epoch(model, loss_fn, optim, X, y_onehot, batch_size=64, seed=0):
    losses, accs = [], []
    for Xb, yb in make_batches(X, y_onehot, batch_size=batch_size, shuffle=True, seed=seed):
        logits = model.forward(Xb)
        loss = loss_fn.forward(logits, yb)

        dlogits = loss_fn.backward()
        model.backward(dlogits)
        optim.step(model.capas)

        losses.append(float(loss))
        accs.append(acc_multiclass(logits, yb))

    return float(np.mean(losses)), float(np.mean(accs))


def eval_epoch(model, loss_fn, X, y_onehot, batch_size=256):
    losses, accs = [], []
    for Xb, yb in make_batches(X, y_onehot, batch_size=batch_size, shuffle=False):
        logits = model.forward(Xb)
        loss = loss_fn.forward(logits, yb)
        losses.append(float(loss))
        accs.append(acc_multiclass(logits, yb))
    return float(np.mean(losses)), float(np.mean(accs))


if __name__ == "__main__":
    # 1) Cargar MNIST (70k)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float64)           # (70000, 784)
    y = mnist["target"].astype(np.int64)           # (70000,)

    # 2) Normalizar a [0,1]
    X = X / 255.0

    # 3) Split
    Xtr, ytr, Xva, yva, Xte, yte = train_val_test_split(X, y, seed=42)

    # 4) One-hot
    C = 10
    ytr_oh = one_hot(ytr, C)
    yva_oh = one_hot(yva, C)
    yte_oh = one_hot(yte, C)

    # 5) Modelo (logits C, sin softmax como capa)
    d_in = Xtr.shape[1]
    model = MLP([
        Dense(l2=1e-4, d_in=d_in, d_out=128, activation="relu"), ReLU(),
        Dense(l2=1e-4, d_in=128, d_out=C),  # logits
    ])

    loss_fn = CrossEntropy()
    optim = Adam(lr=1e-3)

    # 6) Train
    epochs = 10
    batch_size = 128

    best_val = float("inf")
    best_snap = None
    patience = 3
    bad = 0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, loss_fn, optim, Xtr, ytr_oh, batch_size=batch_size, seed=ep)
        va_loss, va_acc = eval_epoch(model, loss_fn, Xva, yva_oh)

        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            bad = 0
            best_snap = []
            for layer in model.capas:
                if hasattr(layer, "W"):
                    best_snap.append((layer.W.copy(), layer.b.copy()))
                else:
                    best_snap.append(None)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # 7) Restore best
    if best_snap is not None:
        for layer, snap in zip(model.capas, best_snap):
            if snap is not None and hasattr(layer, "W"):
                W, b = snap
                layer.W[...] = W
                layer.b[...] = b

    # 8) Test
    te_loss, te_acc = eval_epoch(model, loss_fn, Xte, yte_oh)
    print(f"\nTEST | loss {te_loss:.4f} acc {te_acc:.3f}")
