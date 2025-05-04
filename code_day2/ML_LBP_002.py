import cv2
import numpy as np
import os

from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# กำหนดพารามิเตอร์ต่างๆ
# พารามิเตอร์ LBP
RADIUS = 3           # รัศมีรอบพิกเซล
N_POINTS = 8 * RADIUS   # จำนวนจุดรอบพิกเซล
METHOD = 'uniform'   # วิธีคำนวณ LBP (uniform, default, หรืออื่น ๆ)

# พารามิเตอร์ PCA
N_COMPONENTS = 20     # จำนวนคอมโพเนนต์ที่ต้องการลดมิติ

# พารามิเตอร์ SVM
KERNEL = 'rbf'     # เลือกได้เช่น 'linear', 'rbf', 'poly' ฯลฯ
C_VALUE = 1.0         # ค่า C สำหรับ SVM

# ฟังก์ชันสำหรับดึงฟีเจอร์ LBP
def extract_lbp_features(image):
    """
    รับภาพขนาด Gray-scale แล้วคำนวณ LBP Features
    คืนค่า Histogram ของ LBP ที่ถูก Normalize
    """
    # คำนวณ LBP
    lbp = local_binary_pattern(image, N_POINTS, RADIUS, METHOD)

    # สร้าง histogram ของค่า LBP
    hist, _ = np.histogram(
        lbp.ravel(), 
        bins=np.arange(0, N_POINTS + 3), 
        range=(0, N_POINTS + 2)
    )

    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# ฟังก์ชันโหลดชุดข้อมูล
def load_texture_data(data_path):
    class_names = sorted(os.listdir(data_path))  # เรียงรายชื่อคลาส (โฟลเดอร์)
    
    features = []
    labels = []

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(data_path, class_name)
        
        # ตรวจสอบว่าเป็นโฟลเดอร์จริงหรือไม่
        if not os.path.isdir(class_folder):
            continue
        
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            # โหลดภาพเป็น Gray-scale
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # ดึงฟีเจอร์ LBP
                lbp_hist = extract_lbp_features(image)
                features.append(lbp_hist)
                labels.append(label)
    
    return np.array(features), np.array(labels), class_names


DATA_PATH = "./textures"  


X, y, class_names = load_texture_data(DATA_PATH)

# แบ่งชุดข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# ลดมิติด้วย PCA
pca = PCA(n_components=N_COMPONENTS)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# สร้างและฝึกโมเดล SVM
svm = SVC(kernel=KERNEL, C=C_VALUE, random_state=42)
svm.fit(X_train_pca, y_train)

# ทำนายผลและประเมินโมเดล
y_pred = svm.predict(X_test_pca)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
