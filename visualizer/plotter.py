import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter
import json
import ast

def getFreqPlotFromNewExcel(save_path,excel_path=None,df=None):

    assert excel_path or isinstance(df,pd.DataFrame) , ' both excel_path and df can not be null'
    assert not (excel_path and isinstance(df,pd.DataFrame)) , ' can not pass both excel_path and df simulateanously'
    
    if excel_path:
        df=pd.read_excel(excel_path,sheet_name='vol_annotations')

        
    stages=df['stage']
    stages_flattened=[]
    for val in stages:
        if val[0]=='[':
            val = ast.literal_eval(val)
            stages_flattened.extend(val)
        else:
            stages_flattened.append(val)
            
    freq = Counter(stages_flattened)

    values = list(freq.keys())
    counts = list(freq.values())

    plt.figure(figsize=(8, 4))
    bars=plt.bar(values, counts, color='skyblue', edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height),
                ha='center', va='bottom')
    
    plt.xlabel('Stage')
    plt.ylabel('Frequency')
    plt.title('Frequency of Stages in new dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    
    print('plot saved successfully at :',save_path)
    
if __name__=="__main__":
    
    excel_path="d:\\cleaning_GUI_annotated_data\\06_03_2025\\vol_annotations_06_03_2025.xlsx"
    save_path="images\\plots\\frequencyHistogramNewDataset.png"
    
    getFreqPlotFromNewExcel(save_path,excel_path)
    
    with open("jsons\\patient_level\\train_val_split_new_dataset.json",'r') as f:
        indices=json.load(f)
    
    df=pd.read_excel(excel_path,sheet_name='vol_annotations')

    
    train_df=df.iloc[indices['train_indices']]
    val_df=df.iloc[indices['val_indices']]
    
    getFreqPlotFromNewExcel("images\\plots\\frequencyHistogramNewDataset_trainSet.png",df=train_df)
    getFreqPlotFromNewExcel("images\\plots\\frequencyHistogramNewDataset_testSet.png",df=val_df)
    
    