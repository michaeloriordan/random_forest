import random
import numpy as np
from math import sqrt
from collections import Counter

class Node:
    def __init__(self, col=-1, value=None, target_counts=None,
                 pos_branch=None, neg_branch=None):
        self.col = col
        self.value = value
        self.target_counts = target_counts
        self.pos_branch = pos_branch
        self.neg_branch = neg_branch

    def classification(self):
        return self.target_counts.most_common(1)[0][0]

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, max_features=None,
                 min_samples_split=2, min_samples_leaf=1,
                 class_weight=None, max_split_values=100):
        self.root_node = None
        self.max_depth = max_depth
        self.feature_indices = []
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.max_split_values = max_split_values

    def fit(self, X, y):
        if self.max_features is not None:
            self.feature_indices = self.sample_features(X)
            X = X[:, self.feature_indices]

        if self.max_depth is None:
            self.max_depth = -1

        self.class_weight = self.set_class_weight(self.class_weight, y)

        self.root_node = self.build_tree(X, y, self.max_depth)

        return self

    def predict(self, X):
        if self.max_features is not None:
            X = X[:, self.feature_indices]

        y = np.array([self.classify_sample(x, self.root_node) for x in X])

        return y

    def sample_features(self, X):
        if self.max_features == 'sqrt':
            return random.sample(range(X.shape[1]), int(sqrt(X.shape[1])))
        else:
            raise ValueError('Unknown value for max_features')

    def set_class_weight(self, class_weight, y):
        if class_weight is None:
            return {key: 1 for key in set(y)}
        elif class_weight == 'balanced':
            c = Counter(y)
            n_samples = len(y)
            n_classes = len(c)
            weights = [n_samples / (n_classes * n_y) for n_y in c.values()]
            return {key: w for key, w in zip(c.keys(), weights)}
        else:
            raise ValueError('Unknown value for class_weight')

    def entropy(self, y):
        counts = Counter(y)
        proportions = [(count / len(y)) * self.class_weight[key]
                       for key, count in counts.items()]
        s = np.sum([-p * np.log2(p) for p in proportions])
        return s

    def build_tree(self, X, y, depth):
        if len(X) == 0:
            return Node()
        if depth == 0 or len(X) < self.min_samples_split:
            return Node(target_counts=Counter(y))

        current_entropy = self.entropy(y)
        best_reduction = 0

        for col in range(X.shape[1]):
            values = np.unique(X[:, col])
            if len(values) > self.max_split_values:
                percentiles = np.linspace(0, 100, num=self.max_split_values)
                values = np.percentile(values, percentiles)

            pos_rows = np.array([X[:, col] >= v for v in values])

            n_pos = np.sum(pos_rows, axis=1)
            neg_rows = ~pos_rows
            n_neg = len(X) - n_pos

            valid_values = ((n_pos >= self.min_samples_leaf) *
                            (n_neg >= self.min_samples_leaf))

            if valid_values.size == 0:
                continue

            p = n_pos / len(X)

            new_entropy = (p * np.array([self.entropy(y[pr]) for pr in pos_rows]) +
                           (1-p) * np.array([self.entropy(y[nr]) for nr in neg_rows]))

            entropy_reduction = current_entropy - new_entropy

            entropy_reduction = entropy_reduction[valid_values]
            values = values[valid_values]
            pos_rows = pos_rows[valid_values]
            neg_rows = neg_rows[valid_values]

            if entropy_reduction.max() > best_reduction:
                best_reduction = entropy_reduction.max()
                best_col = col
                best_value = values[entropy_reduction.argmax()]
                best_pos_rows = pos_rows[entropy_reduction.argmax()]
                best_neg_rows = pos_rows[entropy_reduction.argmax()]

            #for value in values:
            #    pos_rows = X[:, col] >= value
            #    neg_rows = ~pos_rows

            #    n_pos = np.sum(pos_rows)
            #    n_neg = np.sum(neg_rows)

            #    p = n_pos / len(X)
            #    new_entropy = (p * self.entropy(y[pos_rows]) +
            #                   (1-p) * self.entropy(y[neg_rows]))
            #    entropy_reduction = current_entropy - new_entropy

            #    if (entropy_reduction > best_reduction and
            #        n_pos >= self.min_samples_leaf and
            #        n_neg >= self.min_samples_leaf):
            #        best_reduction = entropy_reduction
            #        best_col = col
            #        best_value = value
            #        best_pos_rows = pos_rows
            #        best_neg_rows = neg_rows

        if best_reduction > 0:
            pos_branch = self.build_tree(X[best_pos_rows], y[best_pos_rows], depth-1)
            neg_branch = self.build_tree(X[best_neg_rows], y[best_neg_rows], depth-1)
            return Node(col=best_col, value=best_value,
                        pos_branch=pos_branch, neg_branch=neg_branch)
        else:
            return Node(target_counts=Counter(y))

    def classify_sample(self, x, node):
        if node.target_counts is not None:
            return node.classification()
        if x[node.col] >= node.value:
            return self.classify_sample(x, node.pos_branch)
        else:
            return self.classify_sample(x, node.neg_branch)
