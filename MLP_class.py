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
    
    
    
 


