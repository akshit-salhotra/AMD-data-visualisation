import os
import pandas as pd

excel_path="excel/vol_annotations_06_03_2025.xlsx"
count=0
df=pd.read_excel(excel_path,sheet_name='vol_annotations')
for val in df['folder_path']:
    if not os.path.isdir(val):
        print(val)
        count+=1

print(count)