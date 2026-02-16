class MLP:
    def __init__(self, layers: list):
        self.capas = layers

    def forward(self, X):
        aux_salida = X
        for capa in self.capas:
            aux_salida = capa.forward(aux_salida)
        return aux_salida
            

    def backward(self, dOut):
        aux_salida = dOut
        for capa in reversed(self.capas):
            aux_salida = capa.backward(aux_salida)
        return aux_salida
