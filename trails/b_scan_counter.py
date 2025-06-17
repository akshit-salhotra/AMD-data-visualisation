import os
import pandas as pd

excel_path='excel/vol_annotations_06_03_2025.xlsx'

df=pd.read_excel(excel_path,sheet_name="vol_annotations")

paths=df['folder_path']

for folder in paths:
    num_scans=len(os.listdir(folder))
    if num_scans!=128 and num_scans!=200:
        print(folder,num_scans)
