import numpy as np
from cvxopt import matrix, solvers

class SVM():
    def __init__(self, C=None, kernel='linear', degree=3, gamma=None, coef0=1):
        '''
        C: regularization parameter. If None, hard margin is used
        w: weight vector
        b: bias term
        support_vector: data points that are in support vector
        alpha: Lagrange multipliers corresponding to the support vectors
        support_vector_labels: labels of support vectors
        K: Gram Matrix
        
        kernel: method of kernel used
        degree: degree of polynomial kernel
        gamma: gamma parameter for polynomial and rbf kernel
        coef0: coefficient for polynomial and sigmoid kernel
        '''
        self.C = C
        self.w = None
        self.b = None
        self.support_vector = None
        self.alpha = None
        self.support_vector_labels = None
        self.K = None
        
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2, degree=3, gamma=1, coef0=1):
        return (gamma * np.dot(x1, x2) + coef0) ** degree

    def rbf_kernel(self, x1, x2, gamma=1):
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    def _compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == 'linear':
                    K[i,j] = self.linear_kernel(X[i], X[j])
                elif self.kernel == 'poly':
                    K[i,j] = self.polynomial_kernel(X[i], X[j], degree=self.degree, gamma=self.gamma, coef0=self.coef0)
                elif self.kernel == 'rbf':
                    K[i,j] = self.rbf_kernel(X[i], X[j], gamma=self.gamma)
                else:
                    raise ValueError(f"Unsupported kernel type: {self.kernel}. Please choose among linear, poly, and rbf")

        return K
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y = y.astype(float)
        y[y==0] = -1
        
        if self.gamma is None:
            self.gamma = 1 / n_features
            
        self.K = self._compute_kernel_matrix(X) * np.outer(y, y)
        
        P = matrix(self.K)
        q = matrix(-np.ones(n_samples))
        
        A = matrix(y, (1, n_samples))
        B = matrix(0, tc='d')
        
        if self.C is None:
            #hard margin. Not ideal, but it's an option
            G = matrix(-np.eye(n_samples))
            h = matrix(np.zeros(n_samples))
        else:
            #soft margin
            G_max = matrix(-np.eye(n_samples))
            h_max = matrix(np.zeros(n_samples))
            
            G_min = matrix(np.eye(n_samples)) 
            h_min = matrix(np.ones(n_samples) * self.C)
            
            G = matrix(np.vstack((G_max, G_min)))
            h = matrix(np.vstack((h_max, h_min)))  
            
        solvers.options['show_progress'] = False
        solutions = solvers.qp(P, q, G, h, A, B)
        
        alphas = np.ravel(solutions['x'])
        
        sv = alphas > 1e-4
        idx = np.arange(len(alphas))[sv]
        self.alpha = alphas[sv]
        self.support_vector = X[sv]
        self.support_vector_labels = y[sv]
        
        
        self.b = 0
        for i in range(len(self.alpha)):
            self.b += self.support_vector_labels[i]
            self.b -= np.sum(self.alpha * self.support_vector_labels * self.K[idx[i], sv])
        self.b /= len(self.alpha)
        
        if self.kernel == 'linear':
            self.w = np.dot(self.alpha * self.support_vector_labels, self.support_vector)
        else:
            self.w = None
        
        
    def predict(self, X):
        if self.support_vector is None:
            return ValueError("model not fitted")
        
        if self.kernel == 'linear' and self.w is not None:
            decision_values = np.dot(X,self.w) + self.b
        else:
#            decision_values = np.zeros(X.shape[0])
            for i in range(len(self.alpha)):
                if self.kernel == 'linear':
                    self.K = self.linear_kernel(X, self.support_vector.T)
                elif self.kernel == 'poly':
                    self.K = (self.gamma * np.dot(X, self.support_vector.T) + self.coef0) ** self.degree
                elif self.kernel == 'rbf':
                    sq_dist = np.sum(X**2, axis=1).reshape(-1,1) + np.sum(self.support_vector**2, axis=1) - 2 * np.dot(X, self.support_vector.T)
                    self.K = np.exp(-self.gamma * sq_dist)
                else:
                    raise ValueError(f"Unsupported kernel type: {self.kernel}")
                
                '''
                if self.kernel == 'linear':
                    decision_values += self.alpha[i] * self.support_vector_labels[i] * self.linear_kernel(X, self.support_vector[i])
                elif self.kernel == 'poly':
                    decision_values += self.alpha[i] * self.support_vector_labels[i] * self.polynomial_kernel(X, self.support_vector[i], degree=self.degree, gamma=self.gamma, coef0=self.coef0)
                elif self.kernel == 'rbf':
                    decision_values += self.alpha[i] * self.support_vector_labels[i] * self.rbf_kernel(X, self.support_vector[i], gamma=self.gamma)
                '''
            decision_values = np.dot(self.K, self.alpha * self.support_vector_labels) + self.b 
            #decision_values += self.b
        return np.sign(decision_values)