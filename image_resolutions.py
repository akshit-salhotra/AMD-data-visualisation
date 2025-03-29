import os 
from collections import Counter
data_dir=r"d:\\cleaning_GUI_annotated_data\\Cirrus_OCT_Imaging_Data"
def get_res(data_dir):
    resolutions=[]
    for pt_id in os.listdir(data_dir):
        for laterality in os.listdir(data_dir+os.sep+pt_id):
            for date in os.listdir(data_dir+os.sep+pt_id+os.sep+laterality):
                for time in os.listdir(data_dir+os.sep+pt_id+os.sep+laterality+os.sep+date):
                    for opt in os.listdir(data_dir+os.sep+pt_id+os.sep+laterality+os.sep+date+os.sep+time):
                        for machine in os.listdir(data_dir+os.sep+pt_id+os.sep+laterality+os.sep+date+os.sep+time+os.sep+opt):
                            for resolution in os.listdir(data_dir+os.sep+pt_id+os.sep+laterality+os.sep+date+os.sep+time+os.sep+opt+os.sep+machine):
                                resolutions.append(resolution)
    return resolutions

resolutions=get_res(data_dir)
frequency = Counter(resolutions)

for value, count in frequency.items():
    print(f"{value}: {count}")

# print('machines are :',os.listdir(data_dir+os.sep+pt_id+os.sep+laterality+os.sep+date+os.sep+time+os.sep+opt))