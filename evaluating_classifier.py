import torch
from model.resnet_3d import Resnet18_3D
from dataloader import OCTDataset
import json
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Union
import os


def evaluate_dataset(json_path:str,device:torch.device,model_path:str,excel_path:str,transform: Union[transforms.Transform,transforms.Compose],batch_size:int,num_workers:int,timeout:int,save_path:str)->None:
    with open(json_path,'r') as file:
        paths=json.load(file)
        
    model=Resnet18_3D(num_classes=6,device=device)
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)
        
    model.load_state_dict(torch.load(model_path,map_location=device))
    dataset=OCTDataset(paths['test_path'],excel_path,transform)
    dataloader=DataLoader(dataset,batch_size,shuffle=False,num_workers=num_workers,timeout=timeout)

    labels=[]
    preds=[]
    model.eval()
    with torch.no_grad():
        for image,label in dataloader:
            image=image.to(device)
            label=label.to(device)
            logits=model(image)
            pred=torch.argmax(logits,dim=-1)
            labels.append(label)
            preds.append(pred)
    labels=torch.stack(labels,dim=0).cpu().numpy()
    preds=torch.stack(preds,dim=0).cpu().numpy()

    c_matrix=confusion_matrix(labels,preds)

    plt.figure(figsize=(6, 4))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save as image
    plt.savefig(save_path, dpi=300)

    print('the accuarcy of model is:',np.mean(labels==preds))
    
if __name__=="__main__":
    model_path=None
    device='cuda' if torch.cuda.is_available() else 'cpu'
    json_path="jsons\\train_test_val_split.json"
    excel_path=r"d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])
    batch_size=16
    num_workers=8
    timeout=600
    results_dir="results"
    os.makedirs(results_dir,exist_ok=True)
    save_file= model_path.split(os.sep)[-2]+"_"+model_path.split(os.sep)[-1]+'confusion_matrix_test_set.png'
    
    evaluate_dataset(json_path,device,model_path,excel_path,transform,batch_size,num_workers,timeout,results_dir+os.sep+save_file)