import numpy as np
from nn.layers import Dense
from nn.activation import ReLU, Sigmoid
from nn.losses import BinaryCrossEntropy
from nn.model import MLP

L2_LAMBDA = 0.02  # lo que quieres checkear

def compute_loss(model, loss_fn, X, y):
    """
    Loss total consistente con tu implementación:
      - data_loss = BCE (ya promedia por N)
      - + sum_l (l2/2) * ||W_l||^2  usando layer.l2 (cada Dense)
    """
    y_hat = model.forward(X)
    data_loss = loss_fn.forward(y_hat, y)

    l2 = 0.0
    for layer in model.capas:
        if hasattr(layer, "W") and getattr(layer, "l2", 0.0) > 0:
            l2 += 0.5 * layer.l2 * np.sum(layer.W ** 2)

    return float(data_loss + l2)


def gradient_check(model, loss_fn, X, y, epsilon=1e-5, n_checks=10):
    """
    Check de dW de la primera Dense (layer 0), consistente con:
      - Dense.backward: dW = data_grad + l2*W   (ya incluye L2)
      - Sigmoid.backward: espera dA (dL/da)
      - BCE.forward: loss media por N
    """
    # 1) Gradiente analítico (backprop)
    y_hat = model.forward(X)
    N = X.shape[0]

    # dL/da para BCE (porque tu Sigmoid.backward multiplica por a(1-a))
    dOut = (1.0 / N) * (y_hat - y) / (y_hat * (1.0 - y_hat) + 1e-15)

    model.backward(dOut)

    layer = model.capas[0]
    analytical_grad = layer.dW.copy()  # OJO: ya incluye L2 por cómo está Dense.backward

    # 2) Gradiente numérico (central difference)
    indices = [tuple(idx) for idx in np.random.randint(0, layer.W.shape, (n_checks, 2))]

    print(f"{'Index':<10} | {'Analítico':<15} | {'Numérico':<15} | {'Rel. Error':<15}")
    print("-" * 65)

    errors = []

    for idx in indices:
        old_val = layer.W[idx]

        layer.W[idx] = old_val + epsilon
        loss_plus = compute_loss(model, loss_fn, X, y)

        layer.W[idx] = old_val - epsilon
        loss_minus = compute_loss(model, loss_fn, X, y)

        layer.W[idx] = old_val

        grad_num = (loss_plus - loss_minus) / (2.0 * epsilon)
        grad_ana = analytical_grad[idx]

        rel_error = abs(grad_ana - grad_num) / (abs(grad_ana) + abs(grad_num) + 1e-15)
        errors.append(rel_error)

        print(f"{str(idx):<10} | {grad_ana:15.8e} | {grad_num:15.8e} | {rel_error:15.8e}")

    return float(np.mean(errors))


if __name__ == "__main__":
    np.random.seed(0)
    X_check = np.random.randn(4, 2)
    y_check = np.array([[0], [1], [1], [0]])

    # Importante: tus Dense reciben l2 en el constructor. Así TU red usa L2 internamente.
    model = MLP([Dense(L2_LAMBDA, 2, 4), ReLU(), Dense(L2_LAMBDA, 4, 1), Sigmoid()])
    loss_fn = BinaryCrossEntropy()

    avg_error = gradient_check(model, loss_fn, X_check, y_check, epsilon=1e-5, n_checks=10)

    print(f"\nError relativo promedio: {avg_error:.8e}")

    # Umbral razonable con numérico
    if avg_error < 1e-6:
        print("✅ Gradient Check PASSED (consistente con tu red y L2=0.02).")
    else:
        print("❌ Gradient Check FAILED: revisa derivadas / caches / shapes.")
