from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# โหลดชุดข้อมูล Iris
iris = load_iris()
X = iris.data  # คุณลักษณะ (features)
y = iris.target  # ค่าที่ต้องการทำนาย (labels)

# แบ่งข้อมูลเป็นชุด train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ปรับ scaling ด้วย StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างและฝึกโมเดล KNN
k = 3  # จำนวนเพื่อนบ้านที่ใกล้ที่สุด
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# ทำนายผลลัพธ์
y_pred = knn.predict(X_test)

# ประเมินผลลัพธ์
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# คำนวณ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# แสดงผล Confusion Matrix ในรูปแบบตาราง
print("Confusion Matrix:")
print(cm)

# แสดงผล Confusion Matrix ในรูปแบบภาพ
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
