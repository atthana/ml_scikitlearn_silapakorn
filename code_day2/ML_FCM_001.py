import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# สุ่มข้อมูล 2 มิติ
np.random.seed(42)
data1 = np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2))
data2 = np.random.normal(loc=[6, 6], scale=0.5, size=(100, 2))
data3 = np.random.normal(loc=[4, 8], scale=0.5, size=(100, 2))
data = np.vstack((data1, data2, data3))

# กำหนดจำนวนกลุ่ม (Clusters)
n_clusters = 3

# เรียกใช้ Fuzzy C-Means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T,  # ข้อมูลต้องอยู่ในรูป (features, samples)
    c=n_clusters,  # จำนวนกลุ่ม
    m=2.0,  # ค่า fuzziness
    error=0.005,  # ค่าความคลาดเคลื่อนที่ยอมรับได้
    maxiter=1000,  # จำนวนรอบสูงสุด
    init=None,  # ค่าเริ่มต้น
    seed=42  # กำหนด seed เพื่อผลลัพธ์ที่เหมือนกันทุกครั้ง
)

# การจัดกลุ่มข้อมูลโดยเลือกกลุ่มที่มีค่าสมาชิกสูงสุด
labels = np.argmax(u, axis=0)

# แสดงผลข้อมูลและกลุ่มที่จัดได้
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(cntr[:, 0], cntr[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title('Fuzzy C-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()
