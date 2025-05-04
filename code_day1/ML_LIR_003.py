import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# โหลดชุดข้อมูล
diabetes = load_diabetes()

X = diabetes.data[:, np.newaxis, 8]  # ฟีเจอร์ที่ต้องการ
y = diabetes.target  # เป้าหมาย

# แบ่งข้อมูลเป็นชุด Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# พยากรณ์ค่าด้วยชุด Test
y_pred = model.predict(X_test)

# คำนวณ Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# แสดงผลกราฟ
plt.scatter(X_test, y_test, color="blue", label="Actual Data")  # จุดข้อมูลจริง
plt.plot(X_test, y_pred, color="red", label="Regression Line")  # เส้นพยากรณ์
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Linear Regression on Diabetes Dataset")
plt.legend()
plt.show()
