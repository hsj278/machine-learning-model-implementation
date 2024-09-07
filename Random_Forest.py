import numpy as np
import Decision_Tree

class RandomForest():
    def __init__(self, n_tree=10, max_depth=None, min_sample_split=2, n_features=None, regression=False):
        self.n_tree = n_tree
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.trees = []
        self.regression = regression
        
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_tree):
            tree = Decision_Tree.DecisionTree(max_depth=self.max_depth, min_sample_split=self.min_sample_split, 
                                              n_features=self.n_features, regression=self.regression, random_foest=True)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict_tree(self, tree, X):
        return tree.predict(X)
    
    def predict(self, X):
        tree_preds = np.array([self.predict_tree(tree, X) for tree in self.trees])
        
        if self.regression:
            return np.mean(tree_preds, axis=0)
        else:
            def most_freq_class(preds):
                unique_labels, counts = np.unique(preds, return_counts=True)
                return unique_labels[np.argmax(counts)]
            
            most_freq_classes = np.apply_along_axis(most_freq_class, axis=0, arr=tree_preds)
            return most_freq_classes