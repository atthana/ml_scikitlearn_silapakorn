from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# สร้างข้อมูลจำลอง (2 คลาส)
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)

# สร้างโมเดล SVM
model = SVC(kernel='linear', C=1.0)  # ใช้ Linear Kernel
model.fit(X, y)  # ฝึกโมเดลด้วยข้อมูลที่สร้างขึ้น

# การแสดงผล Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# ทำนายผลบนพื้นที่ตาราง
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# การแสดงผลกราฟ
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)  # แสดง Decision Boundary
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired, s=100)  # จุดข้อมูลจริง
plt.title("SVM with Linear Kernel (Custom Dataset)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
