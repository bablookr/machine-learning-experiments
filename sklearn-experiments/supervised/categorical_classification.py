from sklearn.naive_bayes import CategoricalNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC

def naive_bayes(x_train, y_train):
    model = CategoricalNB()
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def one_vs_one(x_train, y_train):
    model = OneVsOneClassifier(SVC())
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def one_vs_rest(x_train, y_train):
    model = OneVsRestClassifier(SVC())
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score