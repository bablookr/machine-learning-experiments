from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Lasso, LassoCV
import numpy as np


def grid_search(x_train, y_train):
    lasso = Lasso(random_state=0, max_iter=10000)
    param_grid = [{'alphas': np.logspace(-4, -0.5, 30)}]
    n_folds = 5
    model = GridSearchCV(lasso, param_grid, cv=n_folds, refit=False)
    model.fit(x_train, y_train)
    mean_score = model.cv_results_['mean_test_score']
    std_score = model.cv_results_['std_test_score']
    return mean_score, std_score


def k_fold(x_train, y_train):
    alphas = np.logspace(-4, -0.5, 30)
    lassoCV = LassoCV(random_state=0, alphas=alphas, max_iter=10000)
    k_fold = KFold(3)
    scores = []
    for k, (train, test) in enumerate(k_fold.split(x_train, y_train)):
        lassoCV.fit(x_train[train], y_train[train])
        scores.append(lassoCV.score(x_train[test], y_train[test]))
    return scores
