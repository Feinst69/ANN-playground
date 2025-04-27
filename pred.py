# Data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

from functions import *
from MLP_class import MLP
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('data.csv', sep=';')
df_work = df[df['Target'] != 'Enrolled']
df_enrolled = df[df['Target'] == 'Enrolled']

colors = ['red', 'green']

# Selecting the target
y = df_work['Target']
y = y.apply(lambda x: 1 if x == 'Graduate' else 0) # 1 for Graduate and 0 for Dropout

# Selecting the features
folder_path = 'features'
# Get all .txt files in the folder
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
txt_data = {}
for filename in txt_files:
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read lines and strip newline characters
        features = [line.strip().strip(',').strip("'").strip('"') for line in file]

    X = df_work[features]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_enrolled = scaler.fit_transform(df_enrolled[features])
    X_enrolled_T = X_enrolled.T

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Keras Model
    input_dim = X_train.shape[1]
    layers = [16,1]
    activations = ['relu', 'sigmoid']
    model_k = KerasModel(input_dim=input_dim, layers=layers, activations=activations, learning_rate=0.05)
    # Train the model
    model_k.train(X_train, y_train, epochs=500, batch_size=32)
    # Predict
    predictions = model_k.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Précision sur le test :",accuracy )
    # Plot accuracy and loss
    model_k.plot_accuracy()
    model_k.plot_loss()
    # plt.savefig(f'keras_model_trainning_{filename}:{accuracy}.png')

    k_result = model_k.predict(X_enrolled)
    k_result = ['Graduated' if x == 1 else 'Dropout' for x in k_result]
    categories = ['Dropout', 'Graduated']
    counts = (k_result.count('Dropout'), k_result.count('Graduated'))
    # Plot the bar chart
    plt.bar(categories, counts, color=colors)
    plt.title('Prediction with Keras')

    # Add counts on top of the bars
    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
    plt.savefig(f'{filename}_keras_model_prediction.png')

    # MLP_class
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.values.reshape(1, -1)
    y_test = y_test.values.reshape(1, -1)
    # Créer le modèle
    mlp = MLP(input_dim=X_train.shape[0], output_dim=1, hidden_layers=(16,), learning_rate=0.1, n_iter=500)
    # Entraîner le modèle
    history = mlp.fit(X_train, y_train)
    # Tester le modèle
    y_pred_test = mlp.predict(X_test)
    # Évaluer les performances
    accuracy = accuracy_score(y_test.flatten(), y_pred_test.flatten())
    print("Précision sur le test :", accuracy )
    plt.savefig(f'mlp_model_trainning_{accuracy}.png')

    mlp_result = mlp.predict(X_enrolled_T)
    mlp_result = np.where(mlp_result, 'Graduate', 'Dropout')
    unique, counts = np.unique(mlp_result, return_counts='Graduate')

    # Add count on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
    colors = ['red', 'green']
    plt.title('Prediction from the np model')
    plt.bar(unique, counts, color=colors)
    plt.savefig('mlp_model_prediction.png')
