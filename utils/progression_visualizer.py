from utils.histogram import get_amdtype
import json
import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from collections import Counter

excel_path="d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
json_path="jsons/train_test_val_split_without_scar.json"

dataframe=pd.read_excel(excel_path,sheet_name='annotations')
with open(json_path,'r') as f:
    paths=json.load(f)

all_paths=paths['train_path']+paths['test_path']

all_paths=sorted(all_paths,key=lambda x:(x.split("\\")[3],x.split("\\")[4],datetime.strptime(x.split("\\")[5], "%Y%m%d").date()))
patients=sorted(list(set(val.split("\\")[3] for val in all_paths)))

i=0

def patient_info_from_path(path):

    sections=path.split("\\")
    scans_info=[sections[3],sections[4],sections[5]]
    return scans_info

progression={}

for patient in patients:

    progression[patient]={'L':[],'R':[]}

    while True:
        if f'{patient+os.sep}L' in all_paths[i]:
            progression[patient]['L'].extend([get_amdtype(patient_info_from_path(all_paths),dataframe)])
            i+=1
        else:
            break
    
    progression[patient]['L']=list(dict.fromkeys(progression[patient]['L']))

    while True:
        if f'{patient+os.sep}R' in all_paths[i]:
            progression[patient]['R'].extend([get_amdtype(patient_info_from_path(all_paths),dataframe)])
            i+=1
        else:
            break

    progression[patient]['R']=list(dict.fromkeys(progression[patient]['R']))
                
def extract_lists(d):
    """Recursively extract all list values from nested dictionaries."""
    result = []
    for v in d.values():
        if isinstance(v, dict):
            result.extend(extract_lists(v))
        elif isinstance(v, list):
            result.append(v)
    return result

all_lists = extract_lists(progression)
counter = Counter(tuple(lst) for lst in all_lists)

# Optional: convert back to list form for readable output
freq = {list(k): v for k, v in counter.items()}  # not hashable, for display only

# Print result
for k, v in counter.items():
    print(f"{list(k)}: {v}")



    