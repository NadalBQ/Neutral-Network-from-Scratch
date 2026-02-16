import numpy as np

class BinaryCrossEntropy:
    def forward(self, y_hat, y) -> float:
        y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
        Z = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        self.y_cache = y
        self.y_hat_cache = y_hat
        return -(1/y_hat.shape[0]) * np.sum(Z)
    
    def backward(self):
        """
        Devuelve dL/da (derivada respecto a la salida del Sigmoid: a = y_hat)
        Consistente con tu Sigmoid.backward(dA) que multiplica por a(1-a).
        """
        y = self.y_cache
        y_hat = self.y_hat_cache
        N = y_hat.shape[0]

        # dL/da para BCE
        dA = (1.0 / N) * (y_hat - y) / (y_hat * (1.0 - y_hat) + 1e-15)
        return dA
    

class WeightedBinaryCrossEntropy:

    def __init__(self, w0: float = 1.0, w1: float = 1.0):
        self.w0 = float(w0) 
        self.w1 = float(w1)  

    def forward(self, y_hat, y) -> float:
        y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)

        w = self.w1 * y + self.w0 * (1.0 - y)

        per_sample = -w * (y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))
        loss = float(np.mean(per_sample))

        self.y_cache = y
        self.y_hat_cache = y_hat
        self.w_cache = w
        return loss

    def backward(self):
        y = self.y_cache
        y_hat = self.y_hat_cache
        w = self.w_cache
        N = y.shape[0]

        dA = w * (y_hat - y) / (y_hat * (1.0 - y_hat) + 1e-15)
        return dA / N

class CrossEntropy:

    def forward(self, logits, y) -> float:
        Z = logits - np.max(logits, axis=1, keepdims=True) # Estabilización
        exp_z = np.exp(Z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True) 

        eps = 1e-15
        per_sample = -np.sum(y * np.log(probs + eps), axis=1)  
        loss = float(np.mean(per_sample)) 


        self.probs = probs
        self.y = y
        return loss
    
    def backward(self):
        return (self.probs - self.y) / self.y.shape[0]