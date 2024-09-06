import numpy as np

class Node():
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None

class DecisionTree():
    def __init__(self, max_depth = 100, threshold_method = 'avg', regression = False): 
        #max_depth: maximum depth of DT. deeper depth can be useful for many variables, but it can also lead to overfitting
        self.max_depth = max_depth
        self.tree = None
        #method of choosing the threshold in numerical variables
        self.threshold_method = threshold_method
        #regression = False -> classification
        self.regression = regression
        
    def _gini_impurity(self, y):
        #obtaining classes of nodes and the number of each classes to calculate the probability
        classes, counts = np.unique(y, return_counts = True)
        probability = counts/np.sum(counts)
        gini = 1 - np.sum(np.square(probability)) 
        #gini measures how mixed the classes are in a node
        #lower gini = node consists more of one class (0 means only one class is in a node)
        return gini
    
    #MSE for regression metric
    def _mse(self, y):
        return np.mean((y - np.mean(y))**2)

    def _evaluate_split(self, thresholds, X, y, feature, threshold, best_score, best_feature, best_threshold):
        for threshold in thresholds:        
            X_left, X_right, y_left, y_right = self._split_idx(X, y, feature, threshold)
        
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            if self.regression:
                score_left = self._mse(y_left)
                score_right = self._mse(y_right)
            else:
                score_left = self._gini(y_left)
                score_right = self._gini(y_right)
                
            weighted_score = (len(y_left) * score_left + len(y_right) * score_right)/len(y)
            
            if weighted_score < best_score:
                best_score = weighted_score
                best_feature = feature
                best_threshold = threshold
        return best_score, best_feature, best_threshold
    
    def _split_idx(self, X, y, feature, threshold):
        left_idx = np.where(X[:, feature] <= threshold)[0]
        right_idx = np.where(X[:, feature] > threshold)[0]
        return X[left_idx], X[right_idx], y[left_idx], y[right_idx]
    
    def _split(self, X, y):
        best_score = float('inf')
        best_feature = None
        best_threshold = None
        for feature in range(X.shape[1]):
            #selecting the threshold between sorted rows as midpoint
            if self.threshold_method == 'avg':
                sorted_X = np.sort(np.unique(X[:, feature]))
                if len(sorted_X) > 1:
                    thresholds = (sorted_X[:-1] + sorted_X[1:]) / 2
                    best_score, best_feature, best_threshold = self._evaluate_split(thresholds, X, y, feature, best_threshold, best_score, best_feature, best_threshold)
            #simply selecting the threshold as the value in the rows
            elif self.threshold_method == 'unique':
                thresholds = np.unique(X[:, feature])
                if len(thresholds) > 1:
                    best_score, best_feature, best_threshold = self._evaluate_split(thresholds, X, y, feature, best_threshold, best_score, best_feature, best_threshold)
   

        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth = 0):
        #calculates the majority class in the node.
        #if this node ends up being a leaf, this is the final prediction of that node
        if self.regression:
            pred_value = np.mean(y)
        else:
            n_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
            pred_value = np.argmax(n_samples_per_class)
        
        #create a node to be attached 
        node = Node(value = pred_value)
        
        #making sure that the depth does not exceed the predetermined max_depth
        if depth < self.max_depth:
            #figuring out the best threshold to split the tree
            feature, threshold = self._split(X, y)
            if feature is not None:
                X_left, X_right, y_left, y_right = self._split_idx(X, y, feature, threshold)
                #if either of them are 0, it means that the node is pure and it is to become a leaf node
                if len(X_left) > 0 and len(X_right) > 0:
                    node.feature = feature
                    node.threshold = threshold
                    
                    #recursively create the tree
                    node.left= self._build_tree(X_left, y_left, depth + 1)
                    node.right = self._build_tree(X_right, y_right, depth + 1)
        
        return node
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        
    def _predict(self, sample, tree):
        node = tree
        while not node.is_leaf():
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
        
    def predict(self, X):
        return np.array([self._predict(sample, self.tree) for sample in X])