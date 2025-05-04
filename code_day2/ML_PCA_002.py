import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

# โหลดชุดข้อมูลเบาหวาน
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Standardize ข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ใช้ PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# ดู explained variance ratio (บอกว่าแต่ละ PC อธิบายข้อมูลได้กี่เปอร์เซ็นต์)
explained_variance = pca.explained_variance_ratio_

# ดู loadings (ความสำคัญของแต่ละฟีเจอร์ใน PCs)
loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(X.shape[1])], index=data.feature_names)

# แสดงผล
print("Explained Variance Ratio:")
print(explained_variance)
print("\nFeature Loadings:")
print(loadings)

# เลือก PC1 เพื่อบอกลำดับความสำคัญของฟีเจอร์
important_features = loadings["PC1"].abs().sort_values(ascending=False)
print("\nFeatures ordered by importance in PC1:")
print(important_features)
