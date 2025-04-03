from trails import exploring_walk
from utils.histogram import get_amdtype,patient_info_from_path
import pandas as pd
import random
import json
import os 
from tqdm import tqdm
from collections import Counter
csv_path="d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
volumes_path=exploring_walk.get_volume_path(exploring_walk.data_dir)

df=pd.read_excel(csv_path,sheet_name='annotations')
print(df.iloc[0])
valid_volumes=[]
for volume in volumes_path:
    # print(volume.split("\\")[-3])
    if volume.split("\\")[-3]=='512X1024X128':
        scan_info=volume.split("\\")[3:6]
        # print(scan_info)
        if len(df[df['research_id']==int(scan_info[0])][df['laterality']==scan_info[1]]):
            valid_volumes.append(volume)
print('the number of valid volumes are:',len(valid_volumes))
train_volumes=[]
test_volumes=[]
patient_ids=[]
random.seed(20)
scar_scans=0
discarded_wet_scans=0
for volume in tqdm(valid_volumes):
    num=random.random()
    sections=volume.split("\\")
    info=[sections[3],sections[4],sections[5]]
    amd_type=get_amdtype(info,df)
    patient_id=volume.split("\\")[3]
    if amd_type=='scar':
        scar_scans+=1
        continue
    if amd_type=='wet':
        try:
            if Counter(patient_ids)[patient_id]>10:
                discarded_wet_scans+=1
                continue
                
        except IndexError:
            pass

    patient_ids.append(patient_id)
    if num<0.80:
        train_volumes.append(volume)
    else:
        test_volumes.append(volume)

for value in ['train','test']:
    print(f'number of volumes in {value} set are :',len(globals()[f'{value}_volumes']))
print('the number of scans labelled with scars:',scar_scans)
print('the number of scans of wet amd discarded:',discarded_wet_scans)
paths={
    'train_path':train_volumes,
    'test_path':test_volumes}

save_path='train_test_val_split_without_scar.json'
os.makedirs('jsons',exist_ok=True)
with open('jsons'+os.sep+save_path,'w') as f:
    json.dump(paths,f,indent=4)