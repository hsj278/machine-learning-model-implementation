#distance metrics: euclidean, manhattan, minkowski, hamming distance for binary
import numpy as np
from collections import Counter

class KNN():
    def __init__(self, k=3, dist='euclidean', regression=False):
        self.k = k
        self.dist = dist
        self.regression = regression
#        self.weighted = weighted
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def _distance(self, x1, x2):
        if self.dist == 'manhattan':
            return np.sum(np.absolute(x1 - x2))
        else:
            return np.sqrt(np.sum((x1 - x2)**2))
        
    def _get_neighbor(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        
        k_idx = np.argsort(distances)[:self.k]
        k_distances = [distances[i] for i in k_idx]
        
        return k_idx, k_distances

    def _majority_vote(self, k_nearest):
        label_counts = {}
        for label in k_nearest:
            label_counts[label] = label_counts.get(label,0) + 1
            
        max_label = None
        max_votes = -1
        for label, count in label_counts.items():
            if count > max_votes:
                max_votes = count
                max_label = label
        return max_label        
        
    def _predict(self, x):
        k_idx, k_distances = self._get_neighbor(x)
        k_nearest = [self.y_train[i] for i in k_idx]
        
        if self.regression:
            return np.mean(k_nearest)
        else:
            return self._majority_vote(k_nearest)
        
    def predict(self, X):
        preds = []
        for x in X:
            preds.append(self._predict(x))
        return np.array(preds)