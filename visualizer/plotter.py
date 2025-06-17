import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter
import json

def getFreqPlotFromNewExcel(save_path,excel_path=None,df=None):
    
    assert excel_path or df , ' both excel_path and df can not be null'
    assert excel_path and df , ' can not pass both excel_path and df simulateanously'
    
    if excel_path:
        df=pd.read_csv(excel_path)
        
    stages=df['stage']
    stages_flattened=[]
    for val in stages:
        if isinstance(val,list):
            stages_flattened.extend(val)
        else:
            stages_flattened.append(val)
            
    freq = Counter(stages_flattened)

    values = list(freq.keys())
    counts = list(freq.values())

    plt.figure(figsize=(8, 4))
    plt.bar(values, counts, color='skyblue', edgecolor='black')
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
    
    df=pd.read_csv(excel_path)
    
    train_df=df.iloc[indices['train_indices']]
    val_df=df.iloc[indices['val_indices']]
    
    getFreqPlotFromNewExcel("images\\plots\\frequencyHistogramNewDataset_trainSet.png",train_df)
    getFreqPlotFromNewExcel("images\\plots\\frequencyHistogramNewDataset_testSet.png",val_df)
    
    