import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
X,Y = make_blobs(n_samples=1000,n_features=2,centers=[(3,3),(7,7),(4,8)],cluster_std=0.5)
plt.scatter(X[:,0],X[:,1],c = 'red',marker = 'o')
plt.show()
#################################################################
db = DBSCAN(eps = 0.3,min_samples = 5)
db.fit(X)
labels = db.labels_
print(labels)
no_clusters = len(np.unique(labels))-1
print("Number of Clusters = ",no_clusters)
no_noise = np.sum(np.array(labels) == -1,axis = 0)
print("Outliers = ",no_noise)
plt.scatter(X[:,0],X[:,1],c = labels,marker = 'o')
plt.title("DBSCAN clustering")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.grid()
plt.show()