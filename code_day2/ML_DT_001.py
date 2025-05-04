import pandas as pd
import numpy as np
from math import log2

# อ่านข้อมูลจากไฟล์ CSV
data = pd.read_csv('tennis_dt.csv')

# ตรวจสอบข้อมูล
print("Preview of the dataset:")
print(data.head())

# คำนวณ Entropy
def entropy(target_col):
    values, counts = np.unique(target_col, return_counts=True)
    entropy_value = -np.sum([(counts[i] / np.sum(counts)) * log2(counts[i] / np.sum(counts)) for i in range(len(values))])
    return entropy_value

# คำนวณ Information Gain
def information_gain(data, feature, target_col):
    total_entropy = entropy(data[target_col])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data[data[feature] == values[i]][target_col]) for i in range(len(values))])
    gain = total_entropy - weighted_entropy
    return gain

# เลือกฟีเจอร์ที่มี Information Gain สูงสุด
def id3(data, features, target_col, parent=None):
    if len(np.unique(data[target_col])) == 1:
        return np.unique(data[target_col])[0]
    elif len(data) == 0:
        return np.unique(parent[target_col])[np.argmax(np.unique(parent[target_col], return_counts=True)[1])]
    elif len(features) == 0:
        return np.unique(data[target_col])[np.argmax(np.unique(data[target_col], return_counts=True)[1])]
    else:
        parent = data
        gains = [information_gain(data, feature, target_col) for feature in features]
        best_feature = features[np.argmax(gains)]
        tree = {best_feature: {}}
        features = [f for f in features if f != best_feature]
        for value in np.unique(data[best_feature]):
            subtree = id3(data[data[best_feature] == value], features, target_col, parent)
            tree[best_feature][value] = subtree
        return tree

# ฟังก์ชันแสดงโครงสร้างต้นไม้ในรูปแบบข้อความ
def print_tree(tree, indent=""):
    for key, value in tree.items():
        if isinstance(value, dict):
            print(f"{indent}{key}")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    print(f"{indent}├── {sub_key}")
                    print_tree(sub_value, indent + "│   ")
                else:
                    print(f"{indent}├── {sub_key}: {sub_value}")
        else:
            print(f"{indent}{key}: {value}")

# สร้าง Decision Tree
features = list(data.columns[:-1])  # ไม่รวมคอลัมน์เป้าหมาย
target_col = 'play'  # ปรับตามชื่อคอลัมน์ในไฟล์
decision_tree = id3(data, features, target_col)

# แสดงผล Decision Tree
print("Decision Tree:")
print_tree(decision_tree)
