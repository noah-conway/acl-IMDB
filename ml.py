import sklearn
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

def dt_train(X, Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    return clf
def kmeans_train(X):
    neigh = KNeighborsClassifier(n_neighbors=3)
    return neigh.fit(X, Y)


def model_test(X, model):
    return model.predict(X)

def compute_F1(Y, Y_hat):
    return f1_score(Y, Y_hat)

