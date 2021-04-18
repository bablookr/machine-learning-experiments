from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import kernels, GaussianProcessRegressor


def linear_regression(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    weights = model.coef_, model.intercept_
    score = model.score(x_train, y_train)
    return weights, score


def ridge_regression(x_train, y_train):
    model = Ridge(alpha=.5)
    model.fit(x_train, y_train)
    weights = model.coef_, model.intercept_
    score = model.score(x_train, y_train)
    return weights, score


def ridge_regression_with_cross_validation(x_train, y_train):
    model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    model.fit(x_train, y_train)
    weights = model.coef_, model.intercept_
    score = model.score(x_train, y_train)
    return weights, score


def svr(x_train, y_train):
    model = SVR()
    model.fit(x_train, y_train)
    weights = model.coef_, model.intercept_
    score = model.score(x_train, y_train)
    return weights, score


def knn(x_train, y_train):
    model = KNeighborsRegressor(n_neighbors=4)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def decision_tree(x_train, y_train):
    model = DecisionTreeRegressor(random_state=0)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score



def random_forest(x_train, y_train):
    model = RandomForestRegressor(max_depth=2, random_state=0)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def bagging(x_train, y_train):
    model = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def ada_boost(x_train, y_train):
    model = AdaBoostRegressor(random_state=0, n_estimators=100)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def gradient_boost(x_train, y_train):
    model = GradientBoostingRegressor(random_state=0, n_estimators=100)
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def gaussian_process(x_train, y_train):
    model = GaussianProcessRegressor(kernel=kernels.DotProduct())
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score