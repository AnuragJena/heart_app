
import numpy as np
from sklearn.feature_selection import mutual_info_classif

class AdaptiveFeatureAttention:
    def __init__(self):
        self.feature_scores = None

    def fit(self, X, y):
        self.feature_scores = mutual_info_classif(X, y)
        self.feature_scores /= np.max(self.feature_scores)

    def transform(self, X):
        if self.feature_scores is None:
            raise RuntimeError("Feature scores not computed. Call 'fit' before 'transform'.")
        return X * self.feature_scores

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
