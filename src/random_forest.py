from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    return rfc
