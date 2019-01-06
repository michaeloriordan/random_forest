import random
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from DecisionTreeClassifier import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt',
                 min_samples_split=2, min_samples_leaf=1, class_weight=None,
                 n_jobs=1, max_split_values=100):
        self.trees = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.max_split_values = max_split_values

    def fit(self, X, y):
        if self.n_jobs > 1:
            return self.fit_parallel(X, y)
        return self.fit_serial(X, y)

    def predict(self, X):
        y = np.array([tree.predict(X) for tree in self.trees]).T
        y = np.array([Counter(row).most_common(1)[0][0] for row in y])
        return y

    def fit_serial(self, X, y):
        self.trees = [self.fit_tree(X, y) for i in range(self.n_estimators)]
        return self

    def fit_parallel(self, X, y):
        with ProcessPoolExecutor(max_workers=self.n_jobs) as e:
            futures = [e.submit(self.fit_tree, X, y)
                       for i in range(self.n_estimators)]
            self.trees = [f.result() for f in futures]
        return self

    def fit_tree(self, X, y):
        tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                      max_features=self.max_features,
                                      min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_split,
                                      class_weight=self.class_weight,
                                      max_split_values=self.max_split_values)
        return tree.fit(X, y)
