# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# Load wine dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Loadings (Feature importance for each Principal Component)
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f"PC{i+1}" for i in range(X.shape[1])], 
    index=data.feature_names
)

# Rank features by importance in PC1
important_features = loadings["PC1"].abs().sort_values(ascending=False)

# Display results
print("Explained Variance Ratio:")
for i, var in enumerate(explained_variance, start=1):
    print(f"PC{i}: {var:.4f}")

print("\nFeature Importance in PC1 (ranked):")
print(important_features)
