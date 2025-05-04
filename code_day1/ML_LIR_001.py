import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# สร้างข้อมูลตัวอย่าง
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # ตัวแปรต้น
print(X)
y = np.array([1.5, 3.7, 4.0, 5.1, 6.8])     # ตัวแปรตาม
print(y)

# สร้างโมเดล Linear Regression
model = LinearRegression()
model.fit(X, y)

# พยากรณ์ค่า
y_pred = model.predict(X)

# แสดงผล
plt.scatter(X, y, color="blue", label="Actual Data")  # จุดข้อมูลจริง
plt.plot(X, y_pred, color="red", label="Regression Line")  # เส้นพยากรณ์
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.legend()
plt.show()

# แสดงค่าความชันและจุดตัดแกน Y
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
