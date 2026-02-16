import numpy as np

# dataset
from sklearn.datasets import load_breast_cancer

# tu framework
from nn.layers import Dense
from nn.activation import ReLU, Sigmoid
from nn.losses import BinaryCrossEntropy
from nn.optim import Adam
from nn.model import MLP


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


def make_batches(X, y, batch_size=64, shuffle=True, seed=0):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, N, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def acc_binary(y_hat, y):
    pred = (y_hat > 0.5).astype(int)
    return float((pred == y).mean())


def train_epoch(model, loss_fn, optim, X, y, batch_size=64, seed=0):
    losses, accs = [], []

    for Xb, yb in make_batches(X, y, batch_size=batch_size, shuffle=True, seed=seed):
        # forward
        y_hat = model.forward(Xb)
        loss = loss_fn.forward(y_hat, yb)

        # backward
        dA = loss_fn.backward()
        model.backward(dA)

        # update
        optim.step(model.capas)

        losses.append(float(loss))
        accs.append(acc_binary(y_hat, yb))

    return float(np.mean(losses)), float(np.mean(accs))


def eval_epoch(model, loss_fn, X, y, batch_size=256):
    losses, accs = [], []

    for Xb, yb in make_batches(X, y, batch_size=batch_size, shuffle=False):
        y_hat = model.forward(Xb)
        loss = loss_fn.forward(y_hat, yb)

        losses.append(float(loss))
        accs.append(acc_binary(y_hat, yb))

    return float(np.mean(losses)), float(np.mean(accs))


if __name__ == "__main__":
    # 1) Cargar dataset
    data = load_breast_cancer()
    X = data["data"].astype(np.float64)  # (N, d)
    y = data["target"].astype(np.float64).reshape(-1, 1)  # (N, 1)

    # 2) Split
    Xtr, ytr, Xva, yva, Xte, yte = train_val_test_split(X, y, seed=42)

    # 3) Normalizar (fit SOLO en train)
    mu, std = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mu, std)
    Xva = standardize_apply(Xva, mu, std)
    Xte = standardize_apply(Xte, mu, std)

    # 4) Modelo (binario)
    d_in = Xtr.shape[1]
    model = MLP([
        Dense(l2=1e-4, d_in=d_in, d_out=32, activation="relu"), ReLU(),
        Dense(l2=1e-4, d_in=32, d_out=1, activation="sigmoid"), Sigmoid(),
    ])

    # OJO: si tu BCE no tiene clip interno, añade clip en tu BCE.forward
    loss_fn = BinaryCrossEntropy()

    optim = Adam(lr=1e-3)


        # 5) Entrenamiento
    epochs = 50
    batch_size = 64

    best_val = float("inf")
    best_snap = None
    patience = 8
    bad = 0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, loss_fn, optim, Xtr, ytr, batch_size=batch_size, seed=ep)
        va_loss, va_acc = eval_epoch(model, loss_fn, Xva, yva)

        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        # Early stopping (snapshot de pesos)
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

    # Restaurar mejores pesos
    if best_snap is not None:
        for layer, snap in zip(model.capas, best_snap):
            if snap is not None and hasattr(layer, "W"):
                W, b = snap
                layer.W[...] = W
                layer.b[...] = b

    # 6) Test
    te_loss, te_acc = eval_epoch(model, loss_fn, Xte, yte)
    print(f"\nTEST | loss {te_loss:.4f} acc {te_acc:.3f}")




