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
from keras.utils import to_categorical
import numpy as np
from collections import Counter

# Class
class KerasModel:
    def __init__(self, input_dim, layers, activations, optimizer='adam', learning_rate=0.01):
        """
        Initialize the CustomModel class.

        Parameters:
        - input_dim: int, the number of input features.
        - layers: list of int, the number of neurons in each layer.
        - activations: list of str, the activation function for each layer.
        - optimizer: str, the optimizer to use (default is 'adam').
        - learning_rate: float, the learning rate for the optimizer (default is 0.001).
        """
        self.input_dim = input_dim
        self.layers = layers
        self.activations = activations
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        """
        Build the Sequential model based on the specified layers and activations.
        """
        model = Sequential()
        
        # Add the first layer with input dimension
        model.add(Dense(self.layers[0], input_dim=self.input_dim, activation=self.activations[0]))
        
        # Add the remaining layers
        for neurons, activation in zip(self.layers[1:], self.activations[1:]):
            model.add(Dense(neurons, activation=activation))
        
        # Compile the model
        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        else:
            raise ValueError("Currently only 'adam' optimizer is supported.")
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the model on the provided training data.

        Parameters:
        - X_train: array-like, the training data.
        - y_train: array-like, the training labels.
        - epochs: int, the number of epochs to train (default is 10).
        - batch_size: int, the batch size for training (default is 32).
        """
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the provided test data.

        Parameters:
        - X_test: array-like, the test data.
        - y_test: array-like, the test labels.
        """
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        """
        Make predictions on the provided data.

        Parameters:
        - X: array-like, the data to make predictions on.
        """
        predictions = self.model.predict(X)
        return [1 if pred >= 0.5 else 0 for pred in predictions]

    def save_model(self, filename):
        """
        Save the model to a file.

        Parameters:
        - filename: str, the name of the file to save the model to.
        """
        self.model.save(filename)

    def summary(self):
        """
        Print the summary of the model.
        """
        self.model.summary()

    def load_model(self, filename):
        """
        Load a model from a file.

        Parameters:
        - filename: str, the name of the file to load the model from.
        """
        from keras.models import load_model
        self.model = load_model(filename)

    # Ajoute model.plot_accuracy() et model.plot_loss()
    def plot_accuracy(self):
        """
        Plot the accuracy of the model.
        """
        import matplotlib.pyplot as plt
        plt.plot(self.history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

    def plot_loss(self):
        """
        Plot the loss of the model.
        """
        import matplotlib.pyplot as plt
        plt.plot(self.history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

   # function for history
    def get_history(self):
        """
        Return the history of the model.
        """
        return self.history.history
    
    def hello(self):
        print("Hello, I am a KerasModel object!")

# price_column is a str name of the numeric column
def iqr_range_target_filter(df, price_column):
    # Calculate the 25th and 75th percentiles using np.percentile with the updated 'method' argument
    q1 = np.percentile(df[price_column].values, 25, method="linear")
    q3 = np.percentile(df[price_column].values, 75, method="linear")

    iqr_range = q3 - q1
    # Filter DataFrame based on IQR range
    new_df = df[
        (df[price_column] > q1 - iqr_range * 1.5) &
        (df[price_column] < q3 + iqr_range * 1.5)
    ]

    # Debugging information
    print("old number of rows", df.shape[0])
    print("new number of rows", new_df.shape[0])

    return new_df


def compare_feature_lists(*lists):
    """
    # Example usage:
    list1 = ['feature1', 'feature2', 'feature3', 'feature4']
    list2 = ['feature3', 'feature4', 'feature5', 'feature6']
    list3 = ['feature2', 'feature4', 'feature7']

    result = compare_feature_lists(list1, list2, list3)
    # Returns something like:
    # {
    #     1: ['feature1', 'feature5', 'feature6', 'feature7'],
    #     2: ['feature2', 'feature3'],
    #     3: ['feature4']
    # }
    """
    # Flatten the list of lists and count occurrences of each element
    all_elements = [item for sublist in lists for item in sublist]
    element_counts = Counter(all_elements)

    # Create dictionary with count as key and list of features as value
    result = {}
    for item, count in element_counts.items():
        if count not in result:
            result[count] = []
        result[count].append(item)

    return result