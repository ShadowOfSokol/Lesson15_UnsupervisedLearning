import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)
X = np.vstack((np.random.randn(50,2)+[2,2],np.random.randn(50,2)+[7,7]))

plt.scatter(X[:,0], X[:,1], s=50, color='gray')
plt.title("Output data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=200, label='Centroids')
plt.title("Result clusters")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()