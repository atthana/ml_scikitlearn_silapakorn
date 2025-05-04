import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# อ่านข้อมูลจากไฟล์
file_path = 'tennis_knn.csv'
data = pd.read_csv(file_path)

# เข้ารหัสข้อมูลข้อความเป็นตัวเลข
label_encoders = {}
custom_encodings = {
    'outlook': {'sunny': 1, 'cloudy': 2, 'rain': 3},
    'temp': {'hot': 1, 'warm': 2, 'cool': 3},
    'humidity': {'high': 1, 'normal': 2},
    'wind': {'strong': 1, 'weak': 2},
    'play': {'no': 0, 'yes': 1}
}

for column in data.columns:
    le = LabelEncoder()
    data[column] = data[column].map(custom_encodings[column])
    label_encoders[column] = le
    print(f"Encoding for column '{column}':")
    for original, encoded in custom_encodings[column].items():
        print(f"  {original} -> {encoded}")

# แยก features และ target
X = data.drop(columns=['play'])
y = data['play']

# ฟังก์ชันสำหรับการใช้งาน KNN
def knn_with_custom_input(k, custom_input):
    # สร้างโมเดล KNN
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X, y)

    # ทำนายจากอินพุตที่กำหนด
    custom_input_encoded = np.array([[custom_encodings[col][custom_input[col]] for col in X.columns]])
    distances, indices = knn.kneighbors(pd.DataFrame(custom_input_encoded, columns=X.columns), n_neighbors=len(X))

    # แสดงระยะทางสำหรับทุกตัวอย่าง
    print("Distances from custom input to each sample:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"Sample {idx}: Distance = {dist:.2f}")

    custom_prediction = knn.predict(pd.DataFrame(custom_input_encoded, columns=X.columns))
    custom_result = [key for key, value in custom_encodings['play'].items() if value == custom_prediction[0]]

    return custom_result[0]

# ตัวอย่างการใช้งาน
# กำหนดค่า k และอินพุตใหม่
k = 4
custom_input = {
    'outlook': 'sunny',
    'temp': 'cool',
    'humidity': 'normal',
    'wind': 'weak'
}

# แปลงค่าอินพุตใหม่เป็นผลลัพธ์
prediction = knn_with_custom_input(k, custom_input)

print(f"Prediction for custom input {custom_input}: {prediction}")
