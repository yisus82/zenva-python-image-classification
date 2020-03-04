from nearest_neighbor import NearestNeighbor
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

"""
CIFAR-10 Dataset: "Learning Multiple Layers of Features from Tiny Images" Alex Krizhevsky, 2009.
"""


class CIFAR10:
    def __init__(self, data_path):
        """Extracts CIFAR10 data from data_path"""
        file_names = ['data_batch_%d' % i for i in range(1, 6)]
        file_names.append('test_batch')

        X = []
        y = []
        for file_name in file_names:
            with open(data_path + file_name, 'rb') as fin:
                data_dict = pickle.load(fin, encoding='bytes')
            X.append(data_dict[b'data'].ravel())
            y = y + data_dict[b'labels']

        self.X = np.asarray(X).reshape(60000, 32 * 32 * 3)
        self.y = np.asarray(y)

        fin = open(data_path + 'batches.meta', 'rb')
        self.LABEL_NAMES = pickle.load(fin, encoding='utf-8')['label_names']
        fin.close()

    def train_test_split(self):
        X_train = self.X[:50000]
        y_train = self.y[:50000]
        X_test = self.X[50000:]
        y_test = self.y[50000:]

        return X_train, y_train, X_test, y_test

    def all_data(self):
        return self.X, self.y

    def __prep_img(self, idx):
        img = self.X[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def show_img(self, idx):
        cv2.imshow(self.LABEL_NAMES[self.y[idx]], self.__prep_img(idx))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_examples(self):
        fig, axes = plt.subplots(5, 5)
        fig.tight_layout()
        for i in range(5):
            for j in range(5):
                rand = np.random.choice(range(self.X.shape[0]))
                axes[i][j].set_axis_off()
                axes[i][j].imshow(self.__prep_img(rand))
                axes[i][j].set_title(self.LABEL_NAMES[self.y[rand]])
        plt.show()


if __name__ == "__main__":
    dataset = CIFAR10('./cifar-10-batches-py/')
    X_train, y_train, X_test, y_test = dataset.train_test_split()
    X, y = dataset.all_data()
    dataset.show_examples()

    """
    nn = NearestNeighbor()
    print("Training...")
    nn.train(X_train, y_train)
    print("Predicting...")
    y_pred = nn.predict(X_test[:100])
    print("Calculating accuracy...")
    accuracy = np.mean(y_test[:100] == y_pred)
    print("Accuracy =", accuracy)
    """

    knn = KNeighborsClassifier(n_neighbors=5, p=1, n_jobs=-1)
    print("Training...")
    knn.fit(X_train, y_train)
    print("Predicting...")
    y_pred = knn.predict(X_test)
    print("Calculating accuracy...")
    accuracy = np.mean(y_test == y_pred)
    print("Accuracy =", accuracy)

    """
    param_grid = {'n_neighbors': [1, 3, 5, 10, 20, 50, 100], 'p': [1, 2]}
    grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    """
