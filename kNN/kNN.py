import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k:int):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        dist = np.apply_along_axis(lambda x: np.apply_along_axis(self.count_distance, 1, self.X, x), 1, X)
        y = self.y[np.apply_along_axis(lambda x: x.argsort()[:self.k], 1, dist)]
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], 1, y)

    def count_distance(self, x, y):
        return np.sqrt(np.sum((x-y)**2))
