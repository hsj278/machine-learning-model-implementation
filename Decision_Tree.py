import numpy as np

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class DecisionTree():
    def __init__(self, max_depth=None, min_sample_split=2, regression=False, threshold_method='unique'):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.root = None
        self.regression = regression
        self.threshold_method = threshold_method
    
    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob = counts/counts.sum()
        return 1 - np.sum(prob**2)
    
    def _mse(self, y):
        return np.mean((y - np.mean(y))**2)
    
    def split_dataset(self, X, y, feature, threshold):
        left_idx = np.where(X[:, feature] <= threshold)[0]
        right_idx = np.where(X[:, feature] > threshold)[0]
        return left_idx, right_idx
    
    def best_split(self, X, y):
        best_score = float('inf')
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            if self.threshold_method == 'avg':
                thresholds = np.sort(np.unique(X[:,feature]))
                thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            else:
                thresholds = np.unique(X[:,feature])
            for threshold in thresholds:
                left_idx, right_idx = self.split_dataset(X, y, feature, threshold)
                if len(left_idx) > 0 and len(right_idx) > 0:
                    if self.regression: #regression score
                        score_left = self._mse(y[left_idx])
                        score_right = self._mse(y[right_idx])
                        weighted_score = (len(left_idx)/n_samples) * score_left + (len(right_idx)/n_samples) * score_right    
                    else: #classification score
                        score_left = self._gini(y[left_idx])
                        score_right = self._gini(y[right_idx])
                        weighted_score = (len(left_idx)/n_samples) * score_left + (len(right_idx)/n_samples) * score_right
                        
                    if weighted_score < best_score:
                        best_score = weighted_score
                        best_feature = feature
                        best_threshold = threshold
                        
        return best_feature, best_threshold
    
    def most_common_label(self, y):
        return np.bincount(y).argmax()
    
    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if self.regression:
            pred_value = np.mean(y)
        else:
#            pred_value = self.most_common_label(y)
            pred_value = np.bincount(y).argmax()
                    
        if (depth == self.max_depth or n_samples < self.min_sample_split or len(np.unique(y)) == 1):
            return Node(value = pred_value)
    
        feature, threshold = self.best_split(X,y)
        if feature is None:
            return Node(value=pred_value)

        left_idx, right_idx = self.split_dataset(X, y, feature, threshold)
        left_subtree = self.build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_subtree = self.build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)
    
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        
    def predict_sample(self, sample, node):
        if node.is_leaf():
            return node.value
        if sample[node.feature] <= node.threshold:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)
        
    def predict(self, X):
        return np.array([self.predict_sample(sample, self.root) for sample in X])
