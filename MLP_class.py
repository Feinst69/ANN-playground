import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

class MLP:
    def __init__(self, input_dim, output_dim, hidden_layers=(16, 16, 16), learning_rate=0.001, n_iter=3000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.dimensions = [input_dim] + list(hidden_layers) + [output_dim]
        self.parametres = self._initialisation()

    def _initialisation(self):
        parametres = {}
        C = len(self.dimensions)
        np.random.seed(1)

        for c in range(1, C):
            parametres['W' + str(c)] = np.random.randn(self.dimensions[c], self.dimensions[c - 1])
            parametres['b' + str(c)] = np.random.randn(self.dimensions[c], 1)

        return parametres
    
    def _forward_propagation(self, X):
        activations = {'A0': X}
        C = len(self.parametres) // 2

        for c in range(1, C + 1):
            Z = self.parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + self.parametres['b' + str(c)]
            if c < C:  # ReLU pour toutes les couches sauf la dernière
                activations['A' + str(c)] = np.maximum(0, Z)
            else:  # Dernière couche : sigmoïde (classification binaire)
                activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

        return activations
    
    def _back_propagation(self, y, activations):
        m = y.shape[1]
        C = len(self.parametres) // 2

        dZ = activations['A' + str(C)] - y
        gradients = {}

        for c in reversed(range(1, C + 1)):
            gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
            gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(self.parametres['W' + str(c)].T, dZ) * (activations['A' + str(c - 1)] > 0)

        return gradients

    


