from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def k_means(x_train):
    model = KMeans(n_clusters=2, random_state=0)
    model.fit(x_train)
    labels = model.labels_
    score = model.score(x_train)
    return labels, score


def gmm(x_train):
    model = GaussianMixture(n_components=2, covariance_type='full')
    model.fit(x_train)
    labels = model.predict(x_train)
    score = model.score(x_train)
    return labels, score