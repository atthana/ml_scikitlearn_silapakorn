import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# สร้างชุดข้อมูลตัวอย่าง
# ตั้ง seed เพื่อให้สุ่มซ้ำได้ผลเดิม (เพื่อการทดลอง)
np.random.seed(42)

# สร้าง DataFrame ด้วยค่าฟีเจอร์ที่เป็นตัวเลข
num_samples = 30
df = pd.DataFrame({
    # สร้างค่า Blood Sugar Level ตั้งแต่ 80 - 160
    'Blood Sugar Level': np.random.randint(80, 161, size=num_samples),
    # สร้างค่า Blood Pressure ตั้งแต่ 70 - 110
    'Blood Pressure': np.random.randint(70, 111, size=num_samples),
    # สร้างค่า BMI ตั้งแต่ 18 - 35
    'BMI': np.random.randint(18, 36, size=num_samples),
    # สร้างค่า Age ตั้งแต่ 20 - 70
    'Age': np.random.randint(20, 71, size=num_samples)
})

# สร้างฟีเจอร์ Exercise Level โดยสุ่มเป็น High / Medium / Low
exercise_levels = ['High', 'Medium', 'Low']
df['Exercise Level'] = np.random.choice(exercise_levels, size=num_samples)

diabetes = []
for i in range(num_samples):
    # ตัวอย่างเงื่อนไขสุ่มเพื่อให้ 'Yes' หรือ 'No'
    sugar = df.loc[i, 'Blood Sugar Level']
    bp = df.loc[i, 'Blood Pressure']
    age = df.loc[i, 'Age']
    rand_factor = np.random.rand()

    # ตัวอย่างเงื่อนไข: ถ้า Blood Sugar > 120 + Age > 50 + rand < 0.8 => Yes
    if (sugar > 120) and (age > 50) and (rand_factor < 0.8):
        diabetes.append('Yes')
    # หรือถ้า BMI > 28 + Blood Pressure > 90 + rand < 0.7 => Yes
    elif (df.loc[i, 'BMI'] > 28) and (bp > 90) and (rand_factor < 0.7):
        diabetes.append('Yes')
    else:
        diabetes.append('No')

df['Diabetes'] = diabetes

# ดูตัวอย่างข้อมูล
print("Sample data:\n", df.head(), "\n")

# แปลงข้อมูล Categorical => Numeric (Label Encoding)
le_exercise = LabelEncoder()
le_diabetes = LabelEncoder()

df['Exercise Level'] = le_exercise.fit_transform(df['Exercise Level'])  # High=0, Medium=1, Low=2 (เรียงตามค่าที่ Encoder กำหนด)
df['Diabetes'] = le_diabetes.fit_transform(df['Diabetes'])  # No=0, Yes=1 (เรียงตามค่าที่ Encoder กำหนด)

# แยก Features (X) และ Target (y)
X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

# สร้างโมเดล Decision Tree และปรับพารามิเตอร์
clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,          # กำหนดความลึกสูงสุด
    min_samples_split=2,  # จำนวนตัวอย่างขั้นต่ำในการแตกโหนด
    min_samples_leaf=1,   # จำนวนตัวอย่างขั้นต่ำในใบ
    random_state=42
)

clf.fit(X, y)

# โครงสร้าง Decision Tree ด้วย scikit-learn (export_text)
tree_rules = export_text(clf, feature_names=X.columns.tolist())

# ฟังก์ชันสำหรับแปลง Decision Tree => Dictionary
def sklearn_to_dict(clf, feature_names):
    from sklearn.tree import _tree
    tree = clf.tree_
    tree_dict = {}

    def recurse(node, current_dict):
        # หากยังไม่เป็น leaf => แตกตามฟีเจอร์ + threshold
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]

            current_dict[f"{name} <= {threshold:.2f}"] = {}
            recurse(tree.children_left[node], current_dict[f"{name} <= {threshold:.2f}"])

            current_dict[f"{name} > {threshold:.2f}"] = {}
            recurse(tree.children_right[node], current_dict[f"{name} > {threshold:.2f}"])
        else:
            # เมื่อเป็น leaf => เก็บค่าคลาสที่โมเดลทำนาย (argmax)
            current_dict["Value"] = tree.value[node].argmax()

    recurse(0, tree_dict)
    return tree_dict

# แปลงเป็น Dictionary และฟังก์ชันแสดงโครงสร้าง Decision Tree
decision_tree_dict = sklearn_to_dict(clf, X.columns.tolist())

def print_tree(tree, indent=""):
    """พิมพ์โครงสร้าง Decision Tree (ในรูป Dictionary) แบบต้นไม้ตามชั้น"""
    for key, value in tree.items():
        if isinstance(value, dict):
            print(f"{indent}{key}")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    # ถ้ายังเป็น dict ย่อย ให้เรียกตนเองเพื่อพิมพ์ต่อ
                    print(f"{indent}├── {sub_key}")
                    print_tree(sub_value, indent + "│   ")
                else:
                    # ถ้าไม่ใช่ dict (leaf แล้ว) ก็พิมพ์ผลลัพธ์
                    print(f"{indent}├── {sub_key}: {sub_value}")
        else:
            print(f"{indent}{key}: {value}")

# แสดงผล Decision Tree (Text Tree) จาก Dictionary
print("\nDecision Tree (Text Tree):")
print_tree(decision_tree_dict)
