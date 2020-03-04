import numpy as np


class NearestNeighbor:
    def __init__(self, distance_func='l1'):
        self.distance_func = distance_func

    def train(self, X, y):
        """X is an N x D matrix such that each row is a training example. y is a N x 1 matrix of true values."""
        self.X_tr = X.astype(np.float32)
        self.y_tr = y

    def predict(self, X):
        """X is an M x D matrix such that each row is a testing example"""
        X_te = X.astype(np.float32)
        num_test_examples = X.shape[0]
        y_pred = np.zeros(num_test_examples, self.y_tr.dtype)

        for i in range(num_test_examples):
            if self.distance_func == 'l2':
                distances = np.sum(np.square(self.X_tr - X_te[i]), axis=1)
            else:
                distances = np.sum(np.abs(self.X_tr - X_te[i]), axis=1)

            smallest_dist_idx = np.argmin(distances)
            y_pred[i] = self.y_tr[smallest_dist_idx]
        return y_pred
