from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def lda(x_train, y_train):
    model = LinearDiscriminantAnalysis()
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score


def qda(x_train, y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    return score