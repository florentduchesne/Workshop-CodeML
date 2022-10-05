import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from dataset_loader import load_dataset, split_dataset

from svm import train_svm
from knn import train_knn
from random_forest import train_random_forest

def visualize(X, Y):
    plt.scatter(range(50), X[:50, 0])
    plt.scatter(range(50), X[50:100, 0])
    plt.scatter(range(50), X[100:, 0])
    plt.show()

def test_modele(modele, X_test, y_test):
    # Predict from the test dataset
    predictions = modele.predict(X_test)

    # Calculate the accuracy
    print(f'Accuracy score : {accuracy_score(y_test, predictions)}')

    # A detailed classification report
    print(classification_report(y_test, predictions))

    X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
    #Prediction of the species from the input vector
    prediction = modele.predict(X_new)
    print("Prediction of Species: {}".format(prediction))

def main():
    X, Y = load_dataset('../iris.csv')
    X_train, X_test, y_train, y_test = split_dataset(X, Y)

    visualize(X, Y)

    svn = train_svm(X_train, y_train)
    
    print('TEST SVM :')
    test_modele(svn, X_test, y_test)

    knn = train_knn(X_train, y_train)

    print('TEST KNN :')
    test_modele(knn, X_test, y_test)

    rfc = train_random_forest(X_train, y_train)

    print('TEST RANDOM FOREST :')
    test_modele(rfc, X_test, y_test)


if __name__ == '__main__':
    main()
