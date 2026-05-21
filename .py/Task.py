import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

data=pd.read_csv("task.csv")
print(data)
X = data.iloc[:, :2].values
X_scaled = StandardScaler().fit_transform(X)

db = DBSCAN(eps = 2,min_samples = 2)
db.fit(X)
labels = db.labels_
print(labels)
no_clusters = len(np.unique(labels))-1
print("Number of Clusters = ",no_clusters)
score = silhouette_score(X, labels)
print("Silhouette Score =", score)
plt.scatter(X[:,0],X[:,1],c = labels,marker = 'o')
plt.title("DBSCAN clustering")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.grid()
plt.show()