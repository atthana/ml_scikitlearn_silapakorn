from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดข้อมูล Iris Dataset
iris = load_iris()
X = iris.data  # ใช้ทั้ง 4 ฟีเจอร์: sepal length, sepal width, petal length, petal width
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
