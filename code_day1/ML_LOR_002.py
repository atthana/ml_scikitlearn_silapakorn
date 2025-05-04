from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# โหลดข้อมูล Iris Dataset
iris = load_iris()
X = iris.data[:, :2]  # ใช้เฉพาะฟีเจอร์ 2 ตัวแรก (sepal length และ sepal width)
y = (iris.target != 0).astype(int)  # แปลงคลาสเป็น binary (0 และ 1)

# แบ่งข้อมูลออกเป็นชุด Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# สร้าง Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)  # ฝึกโมเดลด้วยข้อมูล Train

# ทำนายผล
y_pred = model.predict(X_test)

# คำนวณความแม่นยำ
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# รายงานผลการจำแนก
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# สร้าง Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# การแสดงผลกราฟ
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)  # แสดง Decision Boundary
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired, s=100, label="Data Points")  # จุดข้อมูลจริง
plt.title("Logistic Regression on Iris Dataset (Binary Classification)")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()
plt.show()


#ข้อมูล Iris Dataset:
#ประกอบด้วยข้อมูล 150 ตัวอย่างของดอกกล้วยไม้ 3 ชนิด (Setosa, Versicolor, Virginica)
#ใช้เฉพาะฟีเจอร์ 2 ตัวแรก (sepal length และ sepal width) เพื่อให้ง่ายต่อการแสดงผล
#แปลงคลาสเป็น Binary:
#เปลี่ยนคลาส 3 ชนิดให้เป็นปัญหา Binary Classification
#โดยกำหนดให้คลาส 0 = Setosa และ 1 = ไม่ใช่ Setosa
