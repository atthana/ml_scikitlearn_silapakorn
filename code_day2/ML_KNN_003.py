from sklearn.datasets import load_wine
import pandas as pd

# โหลดข้อมูล
wine = load_wine()

# แปลงข้อมูลเป็น DataFrame
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target

# แสดงตัวอย่างข้อมูล
print(wine_df.head())

#ในชุดข้อมูล Wine Dataset ของ scikit-learn มี 13 คุณลักษณะทางเคมี
#ที่ใช้ในการจำแนกประเภทไวน์ (3 ชนิด) ดังนี้:

#1 Alcohol: ปริมาณแอลกอฮอล์ (%)
#2 Malic acid: ปริมาณกรดมาลิก
#3 Ash: ปริมาณเถ้า (ash)
#4 Alcalinity of ash: ความเป็นด่างของเถ้า
#5 Magnesium: ปริมาณแมกนีเซียม
#6 Total phenols: ปริมาณฟีนอลรวม
#7 Flavanoids: ปริมาณฟลาโวนอยด์
#8 Nonflavanoid phenols: ปริมาณฟีนอลที่ไม่ใช่ฟลาโวนอยด์
#9 Proanthocyanins: ปริมาณโปรแอนโธไซยานิน
#10 Color intensity: ความเข้มของสี
#11 Hue: เฉดสี (Hue)
#12 OD280/OD315 of diluted wines: อัตราส่วนการดูดกลืนแสงที่ 280 nm ต่อ 315 nm ของไวน์ที่เจือจาง
#13 Proline: ปริมาณโปรลีน (กรดอะมิโนชนิดหนึ่ง)
#ประเภทไวน์ (Target Classes):
#ชุดข้อมูลนี้จำแนกประเภทไวน์ออกเป็น 3 ชนิด (หรือคลาส):
#Class 0: ไวน์ชนิดที่ 1
#Class 1: ไวน์ชนิดที่ 2
#Class 2: ไวน์ชนิดที่ 3
