import numpy as np
class SGD:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, layers: list):
        for layer in layers:
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

class Adam:
    def __init__(self, lr : float, beta: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.mW = {}
        self.vW = {}
        self.mb = {}
        self.vb = {}

        
    def step(self, layers: list):
        self.t += 1
        for i, layer in enumerate(layers):
            

            if hasattr(layer, "W"):
                if i not in self.mW:
                    self.mW[i] = np.zeros_like(layer.W)
                    self.vW[i] = np.zeros_like(layer.W)
                    self.mb[i] = np.zeros_like(layer.b)
                    self.vb[i] = np.zeros_like(layer.b)

                gW = layer.dW
                gb = layer.db

                # 1st moment (mean)
                self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * gW
                self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * gb

                # 2nd moment (uncentered variance)
                self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (gW * gW)
                self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (gb * gb)

                # bias correction
                mW_hat = self.mW[i] / (1 - (self.beta1 ** self.t))
                mb_hat = self.mb[i] / (1 - (self.beta1 ** self.t))
                vW_hat = self.vW[i] / (1 - (self.beta2 ** self.t))
                vb_hat = self.vb[i] / (1 - (self.beta2 ** self.t))

                # update
                layer.W -= self.lr * (mW_hat / (np.sqrt(vW_hat) + self.eps))
                layer.b -= self.lr * (mb_hat / (np.sqrt(vb_hat) + self.eps))    
