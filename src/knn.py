from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train):
    nrst_neigh = KNeighborsClassifier(n_neighbors = 3)
    nrst_neigh.fit(X_train, y_train)
    return nrst_neigh
