import numpy as np
from nn.layers import Dense
from nn.activation import ReLU
from nn.activation import Sigmoid
from nn.activation import Tanh

from nn.losses import BinaryCrossEntropy
from nn.optim import SGD
from nn.optim import Adam

from nn.model import MLP
from nn.gradcheck import grad_check_single_weight


def train_step(model, loss_fn, optim, X, y) -> float:
    y_hat = model.forward(X)
    loss = loss_fn.forward(y_hat, y)

    dA = loss_fn.backward()   
    model.backward(dA)

    optim.step(model.capas)
    return float(loss)

# Dataset XOR (batch completo)
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])  # shape (4, 2)

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])  # shape (4, 1)

if __name__ == "__main__":
    model = MLP([Dense(0.01, 2, 4, activation="relu"), ReLU(), Dense(0.01, 4, 1, activation="sigmoid"), Sigmoid()])

    loss_fn = BinaryCrossEntropy()
    optim = Adam(lr=0.1)

    epochs = 3000

    for epoch in range(epochs):
        loss = train_step(model, loss_fn, optim, X, y)

        if epoch % 300 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    y_hat = model.forward(X)
    print("\nPredicciones (probabilidades):")
    print(y_hat)

    print("\nPredicciones binarias:")
    print((y_hat > 0.5).astype(int))




