import numpy as np
from matplotlib import pyplot as plt

class LinearRegression():
    def __init__(self, learning_rate = 0.01, iterations = 1000, fit_method = 'GD', regularization = None, alpha = 0.01, beta = 0.01, init_method = 'random_normal', loss_function = 'MSE'):
        #parameters for gradient descent
        self.learning_rate = learning_rate
        self.iterations = iterations
        
        #method of fitting the model, default is gradient descent
        self.fit_method = fit_method #Choices: 'GD', 'SGD'
        
        #regularization parameters
        self.regularization = regularization  #Choices: 'L1', 'L2', 'EN'
        self.alpha = alpha #Often denoted as lambda
        self.beta = beta #For Elastic Net Regularization

        self.init_method = init_method
        
        self.loss_function = loss_function
        
    def _init_wb(self, n_features):
        #Choices: 'random_normal','random_uniform', 'zeros', 'Xavier', 'normalized_Xavier', 'He_uniform', 'He_Normal'
        #TODO: implement other initialization methods 
        if self.init_method == 'random_normal':
            self.weights = np.random.randn(n_features)
        else:
            self.weights = np.zeros(n_features)
        self.bias = 0
    
    #TODO: implement more loss functions like MAE
    def _compute_loss(self, y, y_pred):
        #Mean Squared Error
        if self.loss_function == 'MSE':
            if self.regularization == 'L1':
                loss = np.mean((y-y_pred)**2) + self.alpha*np.sum(np.absolute(self.weights))
            elif self.regularization == 'L2':
                loss = np.mean((y-y_pred)**2) + self.alpha*np.sum(np.square(self.weights))
            elif self.regularization == 'EN':
                loss = np.mean((y-y_pred)**2) + self.alpha*(self.beta*np.sum(np.absolute(self.weights)) + (1-self.beta)*np.sum(np.square(self.weights)))
            loss = np.mean((y-y_pred)**2)
        #Root Mean Squared Error
        elif self.loss_function == 'RMSE':
            loss = np.sqrt(np.mean((y - y_pred)**2))
        
        return loss
    
    #TODO: Regularization list and loss function list
    
    #TODO: implement other regularization methods like elastic net
    def _apply_regularization(self, gradient):
        if self.regularization == 'L1':
            return gradient + self.alpha * np.sign(self.weights)
        elif self.regularization == 'L2':
            return gradient + self.alpha * self.weights
        elif self.regularization == 'EN':
            return gradient + self.alpha * (self.beta*np.sign(self.weights) + (1 - self.beta)*self.weights)
        return gradient
            
    def GradientDescent(self, X, y):
        n_samples = len(y)
        
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = -(2/n_samples) * np.dot(X.T, (y - y_pred))  
            db = -(2/n_samples) * np.sum(y - y_pred) 
            
            dw = self._apply_regularization(dw)
                    
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            #TODO: save history of the fits to later visualize the iterations
            
        return self.weights, self.bias
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self._init_wb(n_features)
        if self.fit_method == 'GD':
            self.weights, self.bias = self.GradientDescent(X, y)
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias