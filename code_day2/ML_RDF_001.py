import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score

# โหลดข้อมูล Iris
iris = load_iris()
X = iris.data  # คุณลักษณะ (Features)
y = iris.target  # เลเบล (Labels)

# แบ่งข้อมูลเป็นชุดฝึก (Train) และชุดทดสอบ (Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,       # สัดส่วน 30% ใช้เป็นชุดทดสอบ
    random_state=42      # เซ็ตค่าเพื่อให้ผลสุ่มซ้ำได้
)

# สร้างโมเดล RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,    # จำนวนต้นไม้ (ยิ่งมาก โอกาส Overfit น้อย แต่ใช้เวลา compute เพิ่ม)
    random_state=42
)

# ฝึกโมเดลด้วยชุดฝึก
rf_model.fit(X_train, y_train)

# ทำนายชุดทดสอบ
y_pred = rf_model.predict(X_test)

# ประเมินผลด้วย Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)

# คำนวณ Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# แสดงผล Confusion Matrix ด้วย heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
