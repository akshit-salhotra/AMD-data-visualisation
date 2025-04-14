import pandas as pd
import os 
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import json

warnings.filterwarnings('ignore', category=UserWarning)
scans_without_annotations=[]
csv_path="d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
json_path="jsons/train_test_val_split_without_scar.json"
def patient_info(data_dir):
    scans_info=[]
    for pt_id in os.listdir(data_dir):
        for laterality in os.listdir(data_dir+os.sep+pt_id):
            for date in os.listdir(data_dir+os.sep+pt_id+os.sep+laterality):
                scans_info.append([pt_id,laterality,date])
    return scans_info

with open(json_path,'r') as file:
    paths_dict=json.load(file)

# scans_info=patient_info(data_dir)
def patient_info_from_path(paths):
     scans_info=[]
     for path in paths:
          sections=path.split("\\")
          scans_info.append([sections[3],sections[4],sections[5]])
     return scans_info


dataframe=pd.read_excel(csv_path,sheet_name='annotations')
scans_info=patient_info_from_path(paths_dict['train_path'])
AMD_type=[]
count=0


def get_amdtype(info,dataframe):
    date_scan=str(info[-1])
    date_scan=datetime(int(date_scan[:4]),int(date_scan[4:6]),int(date_scan[6:]))
    # print(dataframe.columns)
    filtered=dataframe[(dataframe['research_id']==int(info[0])) ][(dataframe['laterality']==info[1])]
    # print(dataframe.iloc[0])
    # print('info',int(info[0]))
    # print(filtered)
    # print('---')
    try:
        baseline,early,inter,ga,wet,scar,notAMD=filtered[["baseline_stage","Early AMD","Int AMD","GA","Wet","Scar","Not AMD"]].iloc[0]
        dates={'early':early,'inter':inter,'ga':ga,'wet':wet,'scar':scar,'notAMD':notAMD}
        dates = {k: datetime(*list(map(int,v.split('-')))) for k, v in dates.items() if not pd.isna(v)}
        data=dict(sorted(dates.items(),key=lambda x:x[1]))
    except IndexError:
            global count,scans_without_annotations
            count+=1
            print(info)
            scans_without_annotations.append({'patient_id':info[0],'laterality':info[1],'scan_date':info[2]})
            print('count:',count)

    amd_type=None
    for key,val in data.items():
        amd_type=key
        if date_scan<val:
            break
    return amd_type
if __name__=='__main__':
    for info in scans_info:
        amd_type=get_amdtype(info,dataframe) 
        AMD_type.append(amd_type)

    frequency = Counter(AMD_type)

    # Plotting
    plt.bar(frequency.keys(), frequency.values(), color='blue')
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.title('String Frequency Histogram')
    plt.xticks(rotation=45)  # Rotate labels for better visibility
    plt.tight_layout()
    plt.savefig('images/plots/train_data_freq_without_scar_histogram.png')
    plt.show()

    # save_json='scans_without_annotations.json'

    # with open(save_json,'w') as f:
        #  json.dump(scans_without_annotations,f,indent=4)