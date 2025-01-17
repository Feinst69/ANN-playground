# .%%..%%..%%%%%%..%%%%%....%%%%....%%%%...%%...%%...%%%%...%%%%%...%%%%%%..%%.....
# .%%.%%...%%......%%..%%..%%..%%..%%......%%%.%%%..%%..%%..%%..%%..%%......%%.....
# .%%%%....%%%%....%%%%%...%%%%%%...%%%%...%%.%.%%..%%..%%..%%..%%..%%%%....%%.....
# .%%.%%...%%......%%..%%..%%..%%......%%..%%...%%..%%..%%..%%..%%..%%......%%.....
# .%%..%%..%%%%%%..%%..%%..%%..%%...%%%%...%%...%%...%%%%...%%%%%...%%%%%%..%%%%%%.
# .................................................................................

# Libraries
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Class
class KerasModel:
    def __init__(self,
                 input_dim: int,
                 layers: list[tuple[int, str]],
                 learning_rate: float = 0.001):
        """
        Initialize the KerasModel with input dimensions, layer configuration, and learning rate.
        
        :param input_dim: int, number of input features
        :param layers: list of tuples, where each tuple contains the number of neurons and activation function for a layer
        :param learning_rate: float, learning rate for the optimizer
        """
        self.model = Sequential()
        self.model.add(Dense(layers[0][0], input_dim=input_dim, activation=layers[0][1]))
        for neurons, activation in layers[1:]:
            self.model.add(Dense(neurons, activation=activation))
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self,
              X_train: 'array-like',
              y_train: 'array-like',
              epochs: int = 50,
              batch_size: int = 32) -> None:
        """
        Train the model on the provided data.
        
        :param X_train: array-like, training data
        :param y_train: array-like, training labels
        :param epochs: int, number of epochs to train
        :param batch_size: int, size of the batches of data
        """
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self,
                X_test: 'array-like') -> 'array-like':
        """
        Predict using the trained model.
        
        :param X_test: array-like, test data
        :return: array-like, predicted labels
        """
        return self.model.predict(X_test)

    def plot_accuracy(self) -> None:
        """
        Plot the training accuracy over epochs.
        """
        plt.plot(self.history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()

    def plot_loss(self) -> None:
        """
        Plot the training loss over epochs.
        """
        plt.plot(self.history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
