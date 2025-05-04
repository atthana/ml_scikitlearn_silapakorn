import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# อ่านภาพและแปลงเป็น grayscale
image = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)

# พารามิเตอร์สำหรับ LBP
radius = 1  # รัศมีของการพิจารณาพิกเซลรอบ ๆ
n_points = 8 * radius  # จำนวนจุดรอบ ๆ พิกเซลกลาง

# คำนวณ LBP
lbp = local_binary_pattern(image, n_points, radius, method="uniform")

# แสดงผลภาพต้นฉบับและ LBP
plt.figure(figsize=(12, 6))

# ภาพต้นฉบับ
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

# ภาพที่ผ่าน LBP
plt.subplot(1, 2, 2)
plt.title("Local Binary Pattern (LBP)")
plt.imshow(lbp, cmap="gray")
plt.axis("off")

plt.show()
