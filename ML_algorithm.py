import numpy as np
from scipy.special import softmax

import numpy as np

class KNN:
    
    """ K Nearest Neighbors regressor """
    
    def euclidean_distance(self, x_train, x):
        distances = []
        for i, row in x_train.iterrows():            
            err = np.sqrt(np.sum((row - x) ** 2))
            distances.append(err)
        return distances
    
    def __init__(self, k=4):
        self.k = k
        
    def predict(self, x_train, y_train, x_test):
        distances = self.euclidean_distance(x_train, x_test)
        top_k = np.argsort(distances)[:self.k]
        return np.sum(y_train[top_k]) / self.k


class Attention:

    def __init__(self, W_Q, W_K, W_V):
        self.W_Q = np.array(W_Q)
        self.W_K = np.array(W_K)
        self.W_V = np.array(W_V)


    def calculate(self, word, words):
        word = np.array(word)
        words = np.array(words)

        Q = word @ self.W_Q
        K = words @ self.W_K
        V = words @ self.W_V

        weights = Q @ np.transpose(K)
        print("weights b4 softmax =", weights)
        weights = softmax(weights, axis=0)
        print("weights =", weights.shape, weights)
        return weights @ words


class LogisticRegression:

    def __init__(self, W):
        self.W = W

    def forward(self, X):
        logit = X @ self.W
        print("logit =", logit)
        return logit

    def sigmoid(self, logit):
        return np.exp(logit) / (np.exp(logit) + 1)

    def loss(self, X, Y):
        proba = self.calculate(X)
        loss = Y * np.log(proba) + (1 -Y) * np.log(1 - proba)
        return -loss

    def gradient_W(self, X, p, y):
        return x * (p - y)

