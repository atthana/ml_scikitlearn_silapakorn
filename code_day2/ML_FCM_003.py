import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import gaussian

# อ่านภาพและแปลงเป็น Grayscale
image = io.imread('bird.jpg')
image_gray = color.rgb2gray(image)

# ลดขนาดภาพเพื่อประมวลผลเร็วขึ้น
# image_gray = image_gray[::2, ::2]

# ทำ Gaussian Blur เพื่อลด Noise
image_gray = gaussian(image_gray, sigma=1)

# แปลงภาพเป็นเวกเตอร์ 1 มิติ
pixels = image_gray.flatten()

# จำนวนกลุ่ม 
n_clusters = 4

# ใช้ Fuzzy C-Means เพื่อแบ่งกลุ่มพิกเซล
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    np.expand_dims(pixels, axis=0), n_clusters, 2, error=0.005, maxiter=1000, init=None
)

# กำหนดกลุ่มของแต่ละพิกเซล
cluster_membership = np.argmax(u, axis=0)

# แปลงกลับเป็นภาพ (Reshape)
segmented_image = cluster_membership.reshape(image_gray.shape)

# การแสดงผล
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image_gray, cmap='gray')
axes[0].set_title("Original Grayscale Image")
axes[0].axis('off')

axes[1].imshow(segmented_image, cmap='viridis')
axes[1].set_title(f"Segmented Image (k={n_clusters})")
axes[1].axis('off')

plt.tight_layout()
plt.show()
