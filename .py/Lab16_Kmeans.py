import matplotlib.pyplot as plt
from pandas.core.common import random_state
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
###################################################################
X, labels = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=2.75,
    random_state=42
)
###################################################################
kList = []
WCSS = []
x = 16
k = 1
while k != x:
    kmean = KMeans(n_clusters=k, n_init=10, max_iter=300)
    cl = kmean.fit(X)
    centers = cl.cluster_centers_
    print("Centriods = ", centers)
    print("Within Cluster Sum of Squares = ", kmean.inertia_)
    kList.append(k)
    k = k + 1
    WCSS.append(kmean.inertia_)

for i, j in zip(kList, WCSS):
    print(i, "=", j)
plt.plot(kList, WCSS, c = 'g')
plt.xlabel("K")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
# print("Kmean Labels: ", kmean.labels_)
# s = silhouette_score(X, kmean.predict(X))
# print("Silhouette Score = ", s)
# plt.scatter(X[:, 0], X[:, 1], c=kmean.labels_)
# plt.scatter(centers[:, 0], centers[:, 1], c='k', s=200, alpha=0.5)
# plt.show()

