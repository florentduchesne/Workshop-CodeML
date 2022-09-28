# Support vector machine algorithm
from sklearn.svm import SVC

def train_svm(X_train, y_train):
    svn = SVC()
    svn.fit(X_train, y_train)
    return svn
