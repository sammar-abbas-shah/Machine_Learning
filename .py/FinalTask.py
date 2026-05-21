import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

data = pd.read_csv('task.csv')
X = data[['x1', 'x2']].values

db = DBSCAN(eps=3.2, min_samples=2)
db.fit(X)
labels = db.labels_

no_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
no_noise = np.sum(labels == -1)
print("Number of Clusters =", no_clusters)


if no_clusters > 1:
    score = silhouette_score(X, labels)
    print("Silhouette Score =", score)
else:
    print("Not defined.")

plt.scatter(X[:, 0], X[:, 1], c=labels, marker='o')
plt.title("DBSCAN clustering")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.show()
