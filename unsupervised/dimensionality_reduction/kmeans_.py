import numpy as np

class K_Means:

    def __init__(self, X=None, K=None, max_iters=1000, centroids=None):
        self.X = X
        self.K = K
        self.max_iters = max_iters
        self.centroids = centroids if centroids is not None else np.array([])
        np.random.seed(123)
        
    def fit(self, X, K, max_iters):
        self.X = X
        self.K = K
        self.max_iters = max_iters

        if len(self.centroids) == 0:
            # Centroids      
            self.centroids = self.X[np.random.choice(len(self.X), self.K, replace=False)]

        for i in range(self.max_iters):
            # cluster assignment
            distances = np.sqrt(((self.X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            # Recalculation of centroids
            for j in range(self.K):
                self.centroids[j] = np.mean(self.X[self.labels == j], axis=0)

    def transform(self, X):
        distances = []
        for c in self.centroids:
            distances.append(np.linalg.norm(X - c, axis=1))

        distances = np.array(distances)
        return np.argmin(distances, axis=0)
        
    def fit_transform(self, X, K, max_iters):    
        self.fit(X, K, max_iters)
        return self.transform(X)