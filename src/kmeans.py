from dis import dis
import random
import math
import numpy as np

from dataset_loader import load_dataset, split_dataset

class KMoyenne():
    def __init__(self, X, k:int) -> None:
        self.X = X #les donnees brutes
        self.k = k #le nombre de centroides
        self.assignations = np.zeros(X.shape[0]) #le centroide de chaque donnee
        self.distances = np.zeros((X.shape[0], k))
        self.centroides = self.initialiser_centroides(k) #les centroides

    def initialiser_centroides(self, k:int):
        #on choisit k points aleatoires qui seront nos centroides de depart
        return random.choices(self.X, k=k)
    
    def distance(self, p1, p2):
        #distance euclidienne entre deux points
        return np.sum(np.abs(p1 - p2))
    
    def calculer_distances(self):
        #pour chaque point
        for i, x in enumerate(self.X):
            #pour chaque centroide
            for j in range(self.k):
                self.distances[i][j] = self.distance(x, self.centroides[j])
    
    def assigner_centroides(self):
        #pour chaque point
        for i, d in enumerate(self.distances):
            #assigne le centroide le plus proche
            try:
                self.assignations[i] = np.argmin(d)
            except:
                breakpoint()

    def calculer_centroides(self):
        changed = False
        #pour chaque centroide
        for i in range(self.k):
            #recalcule la moyenne du centroide
            new_centroide = np.mean(self.X[np.where(self.assignations == i)], axis=0)
            #si un centroide a change, on continue, sinon on termine l'entrainement
            if self.distance(new_centroide, self.centroides[i]) > 0.1:
                changed = True
            self.centroides[i] = new_centroide
        return changed

    def train(self):
        changed = True
        i = 0
        while changed:
            self.calculer_distances()
            self.assigner_centroides()
            changed = self.calculer_centroides()
            i += 1
            if not changed:
                print(f'A converge apres {i} iterations')

    #evaluer un point
    def evaluate(self, p):
        distances = np.zeros(self.k)
        for i in range(self.k):
            distances[i] = self.distance(p, self.centroides[i])
        return np.argmin(distances)
        


if __name__ == '__main__':
    X, Y = load_dataset('../iris.csv')
    km = KMoyenne(X, 3)
    km.train()
    for i, x in enumerate(X):
        print('i ' + str(km.evaluate(x)))
