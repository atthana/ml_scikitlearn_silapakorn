import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# ฟังก์ชัน Naive Bayes แบบธรรมดา
def naive_bayes_predict(X_train, y_train, new_sample):
    # คำนวณ Prior Probabilities
    priors = y_train.value_counts(normalize=True)

    # คำนวณ Likelihood
    likelihoods = {}
    for col in X_train.columns:
        likelihoods[col] = {}
        for value in X_train[col].unique():
            for class_label in y_train.unique():
                count = ((X_train[col] == value) & (y_train == class_label)).sum()
                total = (y_train == class_label).sum()
                likelihoods[col][(value, class_label)] = count / total

    # คำนวณ Posterior Probabilities
    posteriors = {}
    for class_label in y_train.unique():
        posterior = priors[class_label]
        for col in X_train.columns:
            posterior *= likelihoods[col].get((new_sample[col].iloc[0], class_label), 0)
        posteriors[class_label] = posterior

    # เลือกคลาสที่มีค่า Posterior สูงสุด
    predicted_class = max(posteriors, key=posteriors.get)
    return predicted_class, posteriors

# ตัวอย่างข้อมูลใหม่: "วันนี้ แดดออก อากาศร้อน ความชื้นสูง และ ลมอ่อน"
new_sample = pd.DataFrame([{
    'สภาพอากาศ': 'แดดออก',
    'อุณหภูมิ': 'ร้อน',
    'ความชื้น': 'สูง',
    'ลมพัด': 'อ่อน'
}])

#new_sample = pd.DataFrame([{
#    'สภาพอากาศ': 'แดดออก',
#    'อุณหภูมิ': 'เย็น',
#    'ความชื้น': 'ปกติ',
#    'ลมพัด': 'อ่อน'
#}])

# แปลงข้อมูลใหม่ให้เป็นตัวเลขโดยใช้ Label Encoders เดิม
for column in new_sample.columns:
    new_sample[column] = label_encoders[column].transform(new_sample[column])

# ทำนายผลลัพธ์
predicted_class, posteriors = naive_bayes_predict(X, y, new_sample)

# แปลงผลลัพธ์กลับเป็นข้อความ
predicted_label = label_encoders['class'].inverse_transform([predicted_class])[0]

print("ผลลัพธ์ที่คาดการณ์:", predicted_label)
print("ค่าความน่าจะเป็นสำหรับแต่ละคลาส:")
for class_label, proba in posteriors.items():
    class_name = label_encoders['class'].inverse_transform([class_label])[0]
    print(f"{class_name}: {proba:.4f}")
