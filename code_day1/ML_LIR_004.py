import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# โหลดชุดข้อมูล Diabetes
diabetes = load_diabetes()

# แปลงข้อมูลเป็น DataFrame
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
data['target'] = diabetes.target

# แสดงตัวอย่างข้อมูล
print("ตัวอย่างข้อมูล:")
print(data.head())

# แยกตัวแปรต้น (Features) และตัวแปรเป้าหมาย (Target)
X = data[diabetes.feature_names]  # ฟีเจอร์ทั้งหมด
y = data['target']  # ค่าที่ต้องการพยากรณ์

# แบ่งข้อมูลเป็นชุด Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# พยากรณ์ค่าด้วยชุด Test
y_pred = model.predict(X_test)

# ประเมินผลลัพธ์
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n**ผลลัพธ์การประเมินโมเดล**")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# แสดงค่าความสำคัญของแต่ละฟีเจอร์ (Coefficients)
coefficients = pd.DataFrame({
    "Feature": diabetes.feature_names,
    "Coefficient": model.coef_
})
print("\n**ค่าความสำคัญของฟีเจอร์:**")
print(coefficients)

# สร้างกราฟเปรียบเทียบค่าจริงและค่าที่พยากรณ์
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='black', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2)
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid(True)
plt.show()
