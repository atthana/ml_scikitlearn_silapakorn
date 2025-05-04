import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# โหลดข้อมูล
file_path = 'tennis.csv'
data = pd.read_csv(file_path)

# แปลงข้อมูลที่เป็นข้อความให้เป็นตัวเลข (Encoding)
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# แยก Features (X) และ Target (y)
X = data.drop(columns=['class'])
y = data['class']

# สร้างโมเดล Naive Bayes
model = GaussianNB()
model.fit(X, y)

# ตัวอย่างข้อมูลใหม่: "วันนี้ แดดออก อากาศร้อน ความชื้นสูง และ ลมอ่อน"
new_sample = pd.DataFrame([{
    'สภาพอากาศ': 'แดดออก',
    'อุณหภูมิ': 'ร้อน',
    'ความชื้น': 'สูง',
    'ลมพัด': 'อ่อน'
}])

# แปลงข้อมูลใหม่ให้เป็นตัวเลขโดยใช้ Label Encoders เดิม
for column in new_sample.columns:
    new_sample[column] = label_encoders[column].transform(new_sample[column])

# ทำนายผลลัพธ์
predicted_class = model.predict(new_sample)
predicted_proba = model.predict_proba(new_sample)

# แปลงผลลัพธ์กลับเป็นข้อความ
predicted_label = label_encoders['class'].inverse_transform(predicted_class)

print("ผลลัพธ์ที่คาดการณ์:", predicted_label[0])
print("ค่าความน่าจะเป็นสำหรับแต่ละคลาส:")
for class_name, proba in zip(label_encoders['class'].classes_, predicted_proba[0]):
    print(f"{class_name}: {proba:.4f}")
