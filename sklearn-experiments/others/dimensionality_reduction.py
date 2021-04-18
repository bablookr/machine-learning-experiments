from sklearn.decomposition import PCA


def pca(x_train):
    model = PCA(n_components=2, svd_solver='full')
    model.fit(x_train)
    values = model.singular_values_
    score = model.score(x_train)
    return values, score

