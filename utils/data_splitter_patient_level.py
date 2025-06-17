import pandas as pd
import json
from collections import Counter
import random

def patientLevelDataSplitter(excel_path,save_path,train_split=0.85,tolerance=0.025):
    
    df=pd.read_csv(excel_path)
    # patients=set(df['research_id'])
    frequencies=Counter(list(df['research_id']))
    total_volumes=len(df)
    patients=list(set(k for k in frequencies.keys()))
    # AvgVolsPerPatient=total_volumes/len(patients)
    scansInVal=0
    train_indices=[]   
    val_indices=[]
    
    while scansInVal<total_volumes*(1-train_split):
        sampled = random.sample(patients, k=1)
        scansInVal+=frequencies[sampled]
        patients.remove(sampled)
        val_indices.extend(df.index[df['research_id']==sampled].tolist())
    
    for patient in patients:
        train_indices.extend(df.index[df['research_id']==patient].tolist())
    
    assert len(train_indices)+len(val_indices)==total_volumes, f' the split is not right total volumes:{total_volumes} train_volumes:{len(train_indices)} val_indices:{len(val_indices)}'
    
    assert len(train_indices) <= (total_volumes+tolerance)*train_indices and len(train_indices) >= (total_volumes-tolerance)*train_indices, f' the split has an deviation more than permissible tolerance \n permissible tolerance:{tolerance} \n deviation:{abs(len(train_indices)-total_volumes*train_split)/total_volumes}'
    
    
    with open(save_path,'w') as f:
        json.dump(f,{'train_indices':train_indices
                     ,'val_indices':val_indices})
    print('split saved successfully at ',save_path)
    
    
    
if __name__=="__main__":
    excel_path="d:\\cleaning_GUI_annotated_data\\06_03_2025\\vol_annotations_06_03_2025.xlsx"
    save_path="jsons\\patient_level\\train_val_split_new_dataset.json"
    patientLevelDataSplitter(excel_path,save_path)
    
        