import pandas as pd
import json
excel_path=r"d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"

excel_data = pd.read_excel(excel_path, sheet_name=None)


data_columns_dict={}
# Access sheets as dataframes
for sheet_name, sheet_data in excel_data.items():
    print(f"Sheet Name: {sheet_name}")
    # print(sheet_data.head())  # Display the first few rows\
    print(sheet_data.columns)
    print(sheet_data.iloc[0])
    data_columns_dict[sheet_name]=sheet_data.columns.tolist()

with open('headers.json','w') as file:
    json.dump(data_columns_dict,file,indent=4)