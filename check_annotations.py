import pandas as pd

csv_path="d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"

dataframe=pd.read_excel(csv_path,sheet_name='annotations')

print(dataframe[dataframe['research_id']==737256483])