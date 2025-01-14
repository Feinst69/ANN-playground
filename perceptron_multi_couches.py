import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, n0, n1, n2, learning_rate=0.1, n_iter=1000):
        """
        Paramètres :
        - n0 : nombre d'entrées.
        - n1 : nombre de neurones dans la couche cachée.
        - n2 : nombre de neurones dans la couche de sortie.
        - learning_rate : taux d'apprentissage.
        - n_iter : nombre d'itérations d'entraînement.
        """
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        self.parametres = self.initialisation()
    
    def initialisation(self):
        """Initialise les poids et biais du réseau."""
        W1 = np.random.randn(self.n1, self.n0)
        b1 = np.zeros((self.n1, 1))
        W2 = np.random.randn(self.n2, self.n1)
        b2 = np.zeros((self.n2, 1))

        return {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        } 
    

    def forward_propagation(X, self):
        W1 = self.parametres['W1']
        b1 = self.parametres['b1']
        W2 = self.parametres['W2']
        b2 = self.parametres['b2']

        Z1 = W1.dot(X) + b1
        A1 = 1 / (1 + np.exp(-Z1))

        Z2 = W2.dot(A1) + b2
        A2 = 1 / (1 + np.exp(-Z2))

        return {
            'A1': A1,
            'A2': A2
        }
    
    def back_propagation(self, X, y, activations):
        """Effectue la rétropropagation pour calculer les gradients."""
        A1 = activations['A1']
        A2 = activations['A2']
        W2 = self.parametres['W2']

        m = y.shape[1]

        dZ2 = A2 - y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        return {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
     