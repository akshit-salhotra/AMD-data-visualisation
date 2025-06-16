import torch
# import sys
# sys.path.append(r'D:\AMD-data-visualisation')
# print(sys.path)

from dataloader import OCTDataset,collate_fn
import torchvision.transforms as transforms
import json
from torch.utils.data import DataLoader,Subset
from time import time

if __name__ == "__main__":
    device='cuda' if torch.cuda.is_available() else 'cpu'
    json_path=r"d:\\AMD-data-visualisation\\jsons\\train_test_val_split.json"
    excel_path=r"d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])
    batch_size=1
    num_workers=8
    timeout=600
    # results_dir="results"
    with open(json_path,'r') as file:
            paths=json.load(file)
    # os.makedirs(results_dir,exist_ok=True)
    # save_file= model_path.split(os.sep)[-2]+"_"+model_path.split(os.sep)[-1]+'confusion_matrix_test_set.png'
    new_d=OCTDataset(paths['train_path'],excel_path,transform,undersample=True,multiThread=False)
    new_d=Subset(new_d,range(10))
    new_d=DataLoader(new_d,batch_size=12,shuffle=False,num_workers=num_workers,timeout=timeout,collate_fn=collate_fn)

    t=time()
    for data,_,_ in new_d:
            
        #     print(time()-t)
        #     t=time()
        t=time()