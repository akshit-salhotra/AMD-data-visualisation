import torch
from model.resnet_3d import Resnet18_3D
from dataloader import OCTDataset,collate_fn
import json
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Union
import os
from tqdm import tqdm 
from sklearn.model_selection import KFold

def evaluate_dataset(json_path:str,device:torch.device,model_path:str,excel_path:str,transform,batch_size:int,num_workers:int,timeout:int,save_path:str,data:str)->None:
    with open(json_path,'r') as file:
        paths=json.load(file)
        
    model=Resnet18_3D(num_classes=5).to(device)
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)
    num_folds=5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    dataset=OCTDataset(paths[f'{data}_path'],excel_path,transform,attn=False,undersample=False,classes={'early':0,'inter':1,'ga':2,'wet':3,'notAMD':4})

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):   
        model.load_state_dict(torch.load(model_path,map_location=device))
        dataloader=DataLoader(Subset(dataset,train_idx),batch_size,shuffle=False,num_workers=num_workers,timeout=timeout,collate_fn=collate_fn)

        labels=[]
        preds=[]
        model.eval()
        with torch.no_grad():
            for data,label in tqdm(dataloader):
                # print(len(data))
                data=[d.to(device) for d in data]
                label=label.to(device)
                logits=model(*data)
                # print(logits,label)
                pred=torch.argmax(logits,dim=-1)
                labels.extend(label.detach().cpu().tolist())
                preds.extend(pred.detach().cpu().tolist())
        print(labels,preds)
        # labels=torch.concat(labels,dim=0).cpu().numpy()
        # preds=torch.concat(preds,dim=0).cpu().numpy()
        classes=["early","inter","ga","wet","notamd"]
        c_matrix=confusion_matrix(labels,preds)

        plt.figure(figsize=(6, 4))
        sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=classes,yticklabels=classes)
        plt.title(f'Confusion Matrix,acc:{np.mean(np.array(labels)==np.array(preds)):.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save as image
        plt.savefig(save_path, dpi=300)

        print('the accuarcy of model is:',np.mean(labels==preds))
        break
    
if __name__=="__main__":
    model_path="model_parameter_Resnet\\13\\fold0_epoch24_val_0.2727_train_0.2727"
    device='cuda' if torch.cuda.is_available() else 'cpu'
    json_path="jsons\\train_test_val_split_without_scar.json"
    excel_path=r"d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])
    batch_size=32
    num_workers=12
    timeout=600
    results_dir="results"
    data='train'#can either be train or test
    assert data=='train' or data=='test',"the only permitted values of data are train or test"
    os.makedirs(results_dir,exist_ok=True)
    save_file= model_path.split(os.sep)[-2]+"_"+model_path.split(os.sep)[-1]+f'confusion_matrix_{data}_set_{json_path.split(os.sep)[-1].split(".")[0]}.png'

    evaluate_dataset(json_path,device,model_path,excel_path,transform,batch_size,num_workers,timeout,results_dir+os.sep+save_file,data)