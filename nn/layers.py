import numpy as np

class Dense:

    """
    Capa fully-connected (batch):

      d_out: Numero de neuronas por capa
      d_in: Numero de feutures
      X: (N, d_in)
      W: (d_out, d_in)   # filas = neuronas
      b: (d_out,)
      S = X @ W.T + b    -> (N, d_out)

    """
    
    def __init__(
        self,
        l2: float,
        d_in: int,
        d_out: int,
        seed: int = 0,
        weight_scale: float | None = None,
        activation: str = "relu",
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.l2 = l2
        self.seed = seed

        rng = np.random.default_rng(seed)

        act = activation.lower()

        if act == "relu":
            scale = np.sqrt(2.0 / d_in)
        elif act in ["tanh", "sigmoid"]:
            scale = np.sqrt(2.0 / (d_in + d_out))
        elif act == "small":
            scale = 0.1
        else:
            raise ValueError(f"activation/init desconocido: {activation}")

        if weight_scale is not None:
            scale = weight_scale

        self.W = rng.standard_normal((d_out, d_in)) * scale
        self.b = np.zeros((d_out,), dtype=float)

        self._X_cache: np.ndarray | None = None

    
    def forward(self, X ) -> np.ndarray:

        assert X.ndim == 2, f"X debe ser 2D (N,d). Got: {X.shape}"
        assert X.shape[1] == self.d_in, f"Esperaba d_in={self.d_in}, pero X tiene {X.shape[1]}"

        self._X_cache = X
        S = X @ self.W.T + self.b  # bias broadcast a cada fila
        return S
    

    def backward(self, dS):
        self.dW = (dS.T @ self._X_cache) + self.l2 * self.W
        self.db = np.sum(dS, axis=0)
        return dS @ self.W

    