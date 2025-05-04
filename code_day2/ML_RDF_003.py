import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

# โหลดข้อมูล Digits
digits = load_digits()

# ดึงตัวอย่างของแต่ละตัวเลข (0-9) จากชุดข้อมูล
unique_digits = np.unique(digits.target)  # ตัวเลขที่ไม่ซ้ำ (0-9)
images = []
labels = []

for digit in unique_digits:
    index = np.where(digits.target == digit)[0][0]  # หา index ของตัวเลขแรกที่พบ
    images.append(digits.images[index])  # ดึงภาพของตัวเลขนั้น
    labels.append(digits.target[index])  # เก็บ label (ตัวเลข)

# แสดงผลตัวอย่างรูปตัวเลข (0-9)
plt.figure(figsize=(10, 5))  # กำหนดขนาดหน้าต่าง
for i, (image, label) in enumerate(zip(images, labels)):
    plt.subplot(2, 5, i + 1)  # สร้าง subplot ขนาด 2x5
    plt.imshow(image, cmap='gray')  # แสดงภาพในโหมด grayscale
    plt.title(f"Label: {label}")  # แสดงตัวเลขที่เป็น label
    plt.axis('off')  # ซ่อนแกน

plt.tight_layout()  # ปรับให้ layout ไม่ทับซ้อน
plt.show()
