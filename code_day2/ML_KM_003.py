import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load Wine dataset
wine = load_wine()
X = wine.data  # Features
y = wine.target  # True labels (for reference)

# Standardize the features (สำคัญสำหรับ K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)

# Extract cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Plot the clustering result
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(pca.transform(centers)[:, 0], pca.transform(centers)[:, 1], 
            c='red', marker='X', s=200, label='Cluster Centers')
plt.title("K-Means Clustering on Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
