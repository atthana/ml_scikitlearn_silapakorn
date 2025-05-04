from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# โหลดข้อมูล Iris Dataset
iris = load_iris()
X = iris.data[:, :2]  # ใช้ 2 ฟีเจอร์: sepal length และ sepal width
y = iris.target  # คลาสทั้ง 3 คลาส: Setosa, Versicolor, Virginica

# แบ่งข้อมูลเป็นชุด Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale')  # ใช้ RBF Kernel
model.fit(X_train, y_train)  # ฝึกโมเดลด้วยข้อมูล Train

# ทำนายผล
y_pred = model.predict(X_test)

# คำนวณความแม่นยำ
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# รายงานผลการจำแนก
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# สร้าง Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# การแสดงผล Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.show()

# การแสดงผล Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# ทำนายผลบนพื้นที่ตาราง
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# การแสดงผลกราฟ Decision Boundary พร้อมป้ายชื่อคลาส
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)  # แสดง Decision Boundary
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', cmap=plt.cm.Paired, s=100, label="Test Data Points")  # จุดข้อมูล Test Data

# เพิ่มป้ายชื่อคลาสบนจุด Test Data
for i, txt in enumerate(y_test):
    plt.text(X_test[i, 0] + 0.1, X_test[i, 1] + 0.1, iris.target_names[txt], fontsize=9, color='black')

plt.title("SVM with RBF Kernel (Iris Dataset) - Test Data with Labels")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(loc="upper right")
plt.show()
