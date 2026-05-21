import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# x1, x2 and their corresponding labels y
X = np.array([
    [5, 6],
    [4, 2],
    [5, 4],
    [6, 5],
    [10, 11],
    [11, 8],
    [10, 6],
    [9, 5]
])
y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

loo = LeaveOneOut()
clf = KNeighborsClassifier(n_neighbors=5)

acc = []
for train_idx, test_idx in loo.split(X):
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    acc.append(accuracy_score(y[test_idx], y_pred))

print("Leave‑one‑out accuracy:", np.mean(acc))
