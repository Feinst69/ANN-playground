import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, n0, n1, n2, learning_rate=0.1, n_iter=1000):
        """
        Paramètres :
        - n0 : nombre de neurones dans la couche d'entrée.
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
    
     