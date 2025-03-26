import os
from tqdm import tqdm

data_dir="d:\\cleaning_GUI_annotated_data\\Cirrus_OCT_Imaging_Data"

def get_volume_path(data_dir):
    volumes_path=[]

    for root,dirs,files in tqdm(os.walk(data_dir)):
        rel_path = os.path.relpath(root, data_dir)
        levels = rel_path.split(os.sep)

        if len(levels) == 8:  # pt_id/laterality/date/time/opt/machine
            dirs=[root +os.sep+path for path in dirs]
            volumes_path.extend(dirs)
    return volumes_path

if __name__=="__main__":
    volumes_path=get_volume_path(data_dir)

    print(volumes_path)