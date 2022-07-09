import numpy as np

class KNN():
    
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