import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate elliptical (skewed) 2D data
np.random.seed(42)
mean = [0, 0]
cov = [[3, 1.5], [1.5, 1]]  # Covariance matrix to create an ellipse
data = np.random.multivariate_normal(mean, cov, 200)

# Apply PCA
pca = PCA()
data_pca = pca.fit_transform(data)
principal_components = pca.components_

# Set axis limits for consistent scaling
x_limits = [min(data[:, 0].min(), data_pca[:, 0].min()) - 1,
            max(data[:, 0].max(), data_pca[:, 0].max()) + 1]
y_limits = [min(data[:, 1].min(), data_pca[:, 1].min()) - 1,
            max(data[:, 1].max(), data_pca[:, 1].max()) + 1]

# Plot original and rotated data side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original data
axes[0].scatter(data[:, 0], data[:, 1], alpha=0.7, label="Original Data")
axes[0].quiver(0, 0, principal_components[0, 0], principal_components[0, 1], 
               angles='xy', scale_units='xy', scale=1, color='r', label='PC1')
axes[0].quiver(0, 0, principal_components[1, 0], principal_components[1, 1], 
               angles='xy', scale_units='xy', scale=1, color='b', label='PC2')
axes[0].set_title("Original Data with Principal Components")
axes[0].set_xlabel("X1")
axes[0].set_ylabel("X2")
axes[0].set_xlim(x_limits)
axes[0].set_ylim(y_limits)
axes[0].legend()
axes[0].grid()

# Rotated data (after PCA)
axes[1].scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.7, color="orange", label="Rotated Data")
axes[1].axhline(0, color="gray", linestyle="--")
axes[1].axvline(0, color="gray", linestyle="--")
axes[1].set_title("Rotated Data (PCA Transformation)")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].set_xlim(x_limits)
axes[1].set_ylim(y_limits)
axes[1].legend()
axes[1].grid()

# Display the plots
plt.tight_layout()
plt.show()
