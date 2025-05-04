from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# ข้อมูลตัวอย่าง (X: อุณหภูมิ, y: เปิดหรือไม่เปิดแอร์)
X = np.array([[22], [25], [27], [30], [32], [35], [37], [40]])  # อุณหภูมิ (°C)
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # 1 = เปิดแอร์, 0 = ไม่เปิดแอร์

# แบ่งข้อมูลออกเป็นชุด Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# สร้างและฝึก Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)  # ฝึกโมเดลด้วยชุดข้อมูล Train

# ทำนายผลลัพธ์
y_pred = model.predict(X_test)

# คำนวณความแม่นยำ
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)  # แสดงค่าความแม่นยำของโมเดล

# แสดงผลลัพธ์การทำนาย
print("ผลการทำนาย:", y_pred)

# แสดง Decision Boundary และความน่าจะเป็น
x_range = np.linspace(20, 45, 100).reshape(-1, 1)  # ช่วงของอุณหภูมิ
y_prob = model.predict_proba(x_range)[:, 1]  # ความน่าจะเป็นที่จะเปิดแอร์ (คลาส 1)

# วาดกราฟ
plt.figure(figsize=(8, 6))
plt.scatter(X, y, c=y, cmap='bwr', edgecolor='k', s=100, label='Actual Data')  # แสดงข้อมูลจริง
plt.plot(x_range, y_prob, color='blue', label='Probability (Turn On)')  # แสดงความน่าจะเป็น
plt.axhline(0.5, color='green', linestyle='--', label='Threshold (0.5)')  # เส้น Threshold ที่ 0.5
plt.title("Logistic Regression (Temperature vs. Air Conditioner Usage)")  # หัวข้อกราฟ
plt.xlabel("Temperature (°C)")  # แกน X
plt.ylabel("Probability (Turn On)")  # แกน Y
plt.legend()  # แสดงคำอธิบายกราฟ
plt.show()  # แสดงกราฟ
