import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # True labels (for reference, not used in clustering)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Extract cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot the clustering result
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(pca.transform(centers)[:, 0], pca.transform(centers)[:, 1], 
            c='red', marker='X', s=200, label='Cluster Centers')
plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
