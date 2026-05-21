import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
plt.scatter(X[:,0],X[:,1])
plt.show()
hc = AgglomerativeClustering(n_clusters=5,metric='euclidean',linkage='ward') # average, complete/maximum, single/minimum
y_hc = hc.fit_predict(X)
print("predictions = ",y_hc)
print("Silhoutte score = ",silhouette_score(X,y_hc))
lk = linkage(X,method='ward')
dendrogram(lk)
plt.title("dendrogram")
plt.xlabel("customer")
plt.ylabel("Euclidean Distance ")
plt.show()
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label = "Cluster1")
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='hotpink',label = "Cluster2")
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label = "Cluster3")
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='blue',label = "Cluster4")
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='hotpink',label = "Cluster5")
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k#)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.grid()
plt.show()
