import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import pandas as pd

# สร้างชุดข้อมูลจำลอง
# ตัวแปร: จำนวนการซื้อสินค้า (Purchase Frequency) และยอดใช้จ่ายเฉลี่ย (Average Spending)
data = np.array([
    [10, 1000], [15, 1200], [12, 1100], [30, 3000], [25, 2800],
    [50, 6000], [45, 5500], [60, 6500], [70, 7500], [80, 8000]
])

# Transpose ข้อมูลสำหรับ Fuzzy C-Means
data = data.T

# จำนวนกลุ่มที่ต้องการ (กำหนดให้แบ่งเป็น 3 กลุ่ม)
n_clusters = 3

# ใช้ Fuzzy C-Means ในการจัดกลุ่ม
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data, n_clusters, 2, error=0.005, maxiter=1000, init=None
)

# u (membership matrix): ระดับความเป็นสมาชิกของแต่ละข้อมูลในแต่ละกลุ่ม
cluster_membership = np.argmax(u, axis=0)

# สร้าง DataFrame เพื่อแสดงผลลัพธ์
df = pd.DataFrame({
    'Purchase Frequency': data[0],
    'Average Spending': data[1],
    'Cluster': cluster_membership
})

# การแสดงผลลัพธ์
fig, ax = plt.subplots()
for i in range(n_clusters):
    # เลือกข้อมูลที่อยู่ในกลุ่ม i
    ax.scatter(
        data[0, cluster_membership == i],
        data[1, cluster_membership == i],
        label=f"Cluster {i + 1}"
    )

# แสดงจุดศูนย์กลางของแต่ละกลุ่ม
ax.scatter(cntr[:, 0], cntr[:, 1], c='red', marker='X', s=200, label='Centers')

plt.title("Fuzzy C-Means - Consumer Behavior Clustering")
plt.xlabel("Purchase Frequency")
plt.ylabel("Average Spending")
plt.legend()
plt.grid()
plt.show()

# แสดงตารางข้อมูลในคอนโซล
print("Consumer Behavior Clustering Results:")
print(df)
