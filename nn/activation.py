import numpy as np

class ReLU:
    def forward(self, Z) -> np.ndarray:
        self.Z_cache = Z
        return np.maximum(0.0, Z)
    
    def backward(self, dA):
        dZ = dA * (self.Z_cache > 0)  # máscara booleana -> 0/1 por broadcasting
        return dZ
    

class Sigmoid:

    def forward(self, Z) -> np.ndarray:
        # Sigmoid estable numéricamente (evita overflow en exp)
        A = np.empty_like(Z, dtype=np.float64)

        pos = Z >= 0
        neg = ~pos

        # Z >= 0: exp(-Z) es seguro
        A[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))

        # Z < 0: usa forma equivalente para evitar exp(-Z) enorme
        exp_z = np.exp(Z[neg])          # aquí Z es negativo → exp(Z) no explota
        A[neg] = exp_z / (1.0 + exp_z)

        self.A_cache = A
        return A
    
    # def forward(self, Z) -> np.ndarray:
    #     self.A_cache = 1.0 / (1.0 + np.exp(-Z))
    #     # Los elementos 1.0 realizan broadcasting a todos los elementos de Z y np.esp esta vectorizada
    #     return self.A_cache
    
    def backward(self, dA):
        return dA * (self.A_cache * (1.0 - self.A_cache))
    
class Tanh:
    def forward(self, Z) -> np.ndarray:
        self.A_cache = np.tanh(Z)   
        return self.A_cache
    
    def backward(self, dA):
        return dA * (1.0 - self.A_cache * self.A_cache)





