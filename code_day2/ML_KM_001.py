import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# สร้างข้อมูลตัวอย่าง
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# วาดกราฟข้อมูลต้นฉบับ
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# สร้างโมเดล K-Means
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# ดึงค่า cluster centers และ labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# วาดกราฟผลลัพธ์หลังการจัดกลุ่ม
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# วาดจุดศูนย์กลางของคลัสเตอร์
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-Means Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
