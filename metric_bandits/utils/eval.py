from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def eval_knn(X_train, Y_train, X_test, Y_test, metric):
    """
    Evaluates the performace of metric on a KNN task.
    """
    nbrs = KNeighborsClassifier(n_neighbors=4, algorithm="brute", metric=metric)
    nbrs.fit(X_train, Y_train)

    Y_pred = nbrs.predict(X_test)
    return accuracy_score(Y_test, Y_pred)


def eval_linear(X_train, Y_train, X_test, Y_test):
    """
    Evaluates the performacne of a simple linear classifier
    """
    clf = LinearSVC()
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    return accuracy_score(Y_test, Y_pred)
