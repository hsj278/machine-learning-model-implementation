import numpy as np
from Decision_Tree import DecisionTree
from Linear_Regression import LinearRegression
from Logistic_Regression import LogisticRegression

class AdaBoost():
    def __init__(self, n_estimators=50, max_depth=1, learning_rate=0.1, regression=False, learner='decision_tree'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alphas = [] #weights for each weak learner
        self.models = [] #List of weak learners (Decision stumps)
        self.learning_rate = learning_rate
        self.regression = regression
        self.learner = learner

    def _initialize_learner(self):
        if self.learner == 'decision_tree':
            from Decision_Tree import DecisionTree
            return DecisionTree(max_depth=self.max_depth, regression=self.regression)
        elif self.learner == 'linear_regression':
            from Linear_Regression import LinearRegression
            return LinearRegression()
        elif self.learner == 'logistic_regression':
            from Logistic_Regression import LogisticRegression
            return LogisticRegression()
        elif self.learner == 'KNN':
            from KNN import KNN
            return KNN()
        else:
            raise ValueError(f"Unknown learner: {self.learner}")

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.full(n_samples, (1 / n_samples))
        epsilon = 1e-10
        y_pred = np.zeros(n_samples)

                
        for _ in range(self.n_estimators):
            #Training a weak learner on weighted dataset
            learner = self._initialize_learner()
            #Decision tree, linear regression, KNN
            if self.regression:
                #Calculate error of the stump
                res = y - y_pred
                learner.fit(X, res, sample_weight=w)
                learner_pred = learner.predict(X)
                error_learner = np.mean(w * (res - learner_pred)**2)
                alpha = self.learning_rate * (1-error_learner) 
                
#                y_pred += alpha * learner_pred
                exponent = alpha * np.square(res - learner_pred)
#                exponent = np.clip(exponent, a_min=None, a_max=10)
                w *= np.exp(exponent)
            #Decision tree, logistic regression, KNN
            else:
                error_stump = np.sum(w * (y_pred != y)) / np.sum(w)

                #Calculate the weight of weak learner
                #Amount of say of the stump in the dataset
                alpha = 0.5 * np.log((1 - error_stump + epsilon) / (error_stump + epsilon))
            
                #Update the sample weights
                w *= np.exp(-alpha * y * preds)

            w /= np.sum(w)

            self.models.append(learner)
            self.alphas.append(alpha)
            
    def predict(self, X):
        if self.regression:
            preds = np.zeros(X.shape[0])
            for alpha, model in zip(self.alphas, self.models):
                preds += alpha * model.predict(X)
        else:
            stump_preds = np.array([alpha * model.predict(X) for model, alpha in zip(self.models, self.alphas)])
            preds = np.sign(np.sum(stump_preds, axis = 0))
        return preds