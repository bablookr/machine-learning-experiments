from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV, Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import kernels, GaussianProcessClassifier


def logistic_regression(x_train, y_train):
    model = LogisticRegression(random_state=0)
    model.fit(x_train, y_train)
    weights = model.coef_, model.intercept_
    score = model.score(x_train, y_train)
    return weights, score


def ridge_classifier(x_train, y_train):
    model = RidgeClassifier(tol=1e-2, solver='sag')
    model.fit(x_train, y_train)
    weights = model.coef_, model.intercept_
    score = model.score(x_train, y_train)
    return weights, score


def ridge_classifier_with_cross_validation(x_train, y_train):
    model = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    model.fit(x_train, y_train)
    weights = model.coef_, model.intercept_
    score = model.score(x_train, y_train)
    return weights, score


def perceptron(x_train, y_train):
    model = Perceptron(tol=1e-2)
    model.fit(x_train, y_train)
    weights = model.coef_, model.intercept_
    score = model.score(x_train, y_train)
    return weights, score


def svc(x_train, y_train):
    model = SVC()
    model.fit(x_train, y_train)
    weights = model.coef_, model.intercept_
    score = model.score(x_train, y_train)
    return weights, score


def knn(x_train, y_train):
    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def naive_bayes(x_train, y_train):
    model = BernoulliNB()
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def decision_tree(x_train, y_train):
    model = DecisionTreeClassifier(random_state=0)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def random_forest(x_train, y_train):
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def bagging(x_train, y_train):
    model = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def ada_boost(x_train, y_train):
    model = AdaBoostClassifier(random_state=0, n_estimators=100)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def gradient_boost(x_train, y_train):
    model = GradientBoostingClassifier(random_state=0, n_estimators=100)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def gaussian_process(x_train, y_train):
    model = GaussianProcessClassifier(kernel=kernels.RBF(1.0))
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score