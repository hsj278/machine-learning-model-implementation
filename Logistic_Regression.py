import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate = 0.01, iterations = 1000, fit_method = 'GD', regularization = None, alpha = 0.01, beta = 0.01, init_method = 'random_normal', loss_function = 'BCE'):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.fit_method = fit_method
        
        self.regularization = regularization
        self.alpha = alpha
        self.beta = beta
        
        self.init_method = init_method
        
        self.loss_function = loss_function
        
    def _init_wb(self, n_features):
        if self.init_method == 'random_normal':
            self.weights = np.random.randn(n_features)
        else:
            self.weights = np.zeros(n_features)
        self.bias = 0
        
    def _apply_regularization(self, gradient):
        if self.regularization == 'L1':
            return gradient + self.alpha * np.sign(self.weights)
        elif self.regularization == 'L2':
            return gradient + self.alpha * self.weights
        elif self.regularization == 'EN':
            return gradient + self.alpha * (self.beta * np.sign(self.weights) + (1 - self.beta) * self.weights)
        return gradient
    
    def _compute_loss(self, y, y_pred):
        epsilon = 1e-9

        n_samples = len(y)
        loss = -1/n_samples * np.sum(y * np.log(y_pred + epsilon) +  (1 - y) * np.log(1 - y_pred + epsilon))
    
        if self.regularization == 'L1':
            loss += self.alpha * np.sum(np.abs(self.weights))
        elif self.regularization == 'L2':
            loss += self.alpha * np.sum(np.square(self.weights))
        elif self.regularization == 'EN':
            loss += self.alpha * (self.beta * np.sum(np.abs(self.weights)) + (1 - self.beta) * np.sum(np.sqaure(self.weights)))
            
    def _sigmoid_function(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))
    
    def GradientDescent(self, X, y):
        n_samples = len(y)
        
        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            
            y_pred = self._sigmoid_function(z)
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            dw = self._apply_regularization(dw)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            loss = self._compute_loss(y, y_pred)
            
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_wb(n_features)
        if self.fit_method == 'GD':
            self.GradientDescent(X, y)
        
    def predict(self, X):
        threshold = 0.5
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid_function(z)
        y_pred_thresh = [1 if i > threshold else 0 for i in y_pred]
        
        return np.array(y_pred_thresh)
            