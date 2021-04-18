from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM


def pipeline_1(x_train, y_train):
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', SVC())
    ])
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def pipeline_2(x_train, y_train):
    model = Pipeline(steps=[
        ('rbm', BernoulliRBM(random_state=0)),
        ('classifier', LogisticRegression(tol=1, solver='newton-cg'))
    ])
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score