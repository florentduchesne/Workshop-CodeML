import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def load_dataset(path:str = 'iris.csv'):
    #lit le fichier avec Pandas
    df = pd.read_csv(path)
    
    # Separe les caracteristiques des classes
    data = df.values
    #X et Y sont des arrays numpy
    X = data[:,:4] #les 4 premieres colonnes
    Y = data[:,4] #la derniere colonne

    return X, Y

def split_dataset(X, Y):
    # Separe le dataset entre l'entrainement et les tests
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #20% du dataset sera reserve pour les tests

    return X_train, X_test, y_train, y_test

