import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# โหลดชุดข้อมูล
diabetes = load_diabetes()

print(diabetes.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# age: อายุ (Normalized)
# sex: เพศ (Normalized)
# bmi: ดัชนีมวลกาย (Body Mass Index)
# bp: ความดันโลหิต (Blood Pressure)
# s1: ค่าคอเลสเตอรอลในเลือด (Total Serum Cholesterol)
# s2: ค่าความหนาแน่นของไลโปโปรตีนชนิดต่ำ (Low-Density Lipoproteins - LDL)
# s3: ค่าความหนาแน่นของไลโปโปรตีนชนิดสูง (High-Density Lipoproteins - HDL)
# s4: ค่าความเข้มข้นของไตรกลีเซอไรด์ในเลือด (Triglycerides)
# s5: ค่าความเข้มข้นเฉพาะของน้ำตาลในเลือด (Blood Sugar Level)
# s6: ค่าความดันโลหิตเฉลี่ย (Average Blood Pressure)

#ในชุดข้อมูล diabetes จาก Scikit-learn ค่า target แสดงถึง
#ความก้าวหน้าของโรคเบาหวานในหนึ่งปีหลังจากวัดค่าฟีเจอร์
#(progression of diabetes after one year)
#โดยค่าที่ปรากฏใน target เป็นค่าต่อเนื่อง (Continuous Value)
#ซึ่งสามารถใช้ในการพยากรณ์ด้วยโมเดลแบบ Regression

#target
#ประเภทข้อมูล: เป็นตัวเลขต่อเนื่อง (float)
#ช่วงของค่า: ค่าจะอยู่ในช่วงประมาณ 25 ถึง 346
#ยิ่งค่าสูง หมายถึงโรคเบาหวานมีความรุนแรงมากขึ้นในปีถัดไป
#ค่าน้อย หมายถึงความก้าวหน้าของโรคเบาหวานอยู่ในระดับต่ำ

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print(df)
