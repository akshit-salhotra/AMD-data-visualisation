import torch
import matplotlib
matplotlib.use('Agg')
from model.resnet_3d import Resnet18_3D
from model.resnet_medicalnet import resnet10,resnet34
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
from hooks.batch_hook import create_hook,batchnorm_stats
from utils.make_plots import get_batch_stats_plot
import pandas as pd
from utils.evaluation_metrics import rocPlotter

def evaluate_dataset(json_path:str,device:torch.device,model_path:str,excel_path:str,transform,batch_size:int,num_workers:int,timeout:int,save_path:str,data:str,save_misclassified:bool,excel_save_path:str)->None:
    with open(json_path,'r') as file:
        paths=json.load(file)
    
    n_classes=6
    model=resnet34(num_classes=n_classes,shortcut_type='A').to(device)
    class_dict={'EarlyAMD':0,'Int AMD':1,'GA':2,'Wet':3,'Scar':4,"Not AMD":5}
    class_dict = dict(sorted(class_dict.items(), key=lambda item: item[1]))

    classes=[key for key in class_dict.keys()]
    multilabel=True
    
    assert list(class_dict.values())==sorted(list(class_dict.values())),'the values of class dict keys must be sorted'
    
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)

    dataset_config={'transforms':transform
                        ,'attn':False
                        ,'undersample':True
                        ,'classes':class_dict
                        ,'denoise':False
                        ,'multiThread':False
                        ,'old_excel':False}
    
    dataset=OCTDataset(paths[f'{data}_indices'],excel_path,**dataset_config)
    # dataset=Subset(dataset,range(4))
    model.load_state_dict(torch.load(model_path,map_location=device))
    dataloader=DataLoader(dataset,batch_size,shuffle=False,num_workers=num_workers,timeout=timeout,collate_fn=collate_fn)

    labels=[]
    if multilabel:
        LOGITS=[]
    else:
        preds=[]
    path_scans=[]
    model.eval()
    with torch.no_grad():
        for data,paths,label in tqdm(dataloader):
            data=[d.to(device) for d in data]
            label=label.to(device)
            logits=model(*data)
            
            if multilabel:
                logits=nn.functional.sigmoid(logits)
                LOGITS.extend(logits.detach().cpu().numpy())
            else:
                pred=torch.argmax(logits,dim=-1)
                preds.extend(pred.detach().cpu().numpy())
                
            labels.extend(label.detach().cpu().numpy())     
            if save_misclassified:          
                path_scans.extend(paths)

    if multilabel:
        # print(LOGITS[0].device,label[0].device)
        labels=np.array(labels)

        # print(labels.shape,LOGITS.shape)
        # print([v.shape for v in LOGITS])
        # print([])


        LOGITS=np.array(LOGITS)
        #save_roc= model_path.split(os.sep)[0]+"_"+model_path.split(os.sep)[-2]+"_"+model_path.split(os.sep)[-1]+f'roc_test_set_{(json_path.split(os.sep)[-1]).split(".")[0]}.png'
        save_roc=save_path.replace("confusion_matrix","roc")
        optimalThresholds=rocPlotter(LOGITS,labels,n_classes,save_roc,json_dir=os.path.dirname(model_path),classNames=classes)
        thres=np.array([float(optimalThresholds[key]) for key in classes])
        print(thres.shape,LOGITS.shape)
        preds =(LOGITS >= thres).astype(int)
        conf_matrix_per_class = []
        for i in range(n_classes):
            cm = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1])
            conf_matrix_per_class.append(cm)
            

        # Plot confusion matrix per class in a grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()

        for i in range(n_classes):
            sns.heatmap(conf_matrix_per_class[i],
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        cbar=False,
                        ax=axes[i],
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"])
            axes[i].set_title(f'Class {classes[i]}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

            
    
    else:       
        c_matrix=confusion_matrix(labels,preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=classes,yticklabels=classes)
        plt.title(f'Confusion Matrix,acc:{np.mean(np.array(labels)==np.array(preds)):.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()
    
    if save_misclassified:
        inverse_class_dict={v:k for k,v in class_dict.items()}
        diff_indexes = [i for i, (a, b) in enumerate(zip(preds,labels )) if a != b]
        pred_class=[inverse_class_dict[preds[i]] for i in diff_indexes]
        label_class=[inverse_class_dict[labels[i]] for i in diff_indexes]
        scans=[path_scans[i] for i in diff_indexes]
        df=pd.DataFrame({
            "scan_path":scans,
            "predicted_class":pred_class,
            "actual class":label_class
        })
        df.to_excel(excel_save_path,index=False)



    # get_batch_stats_plot(batchnorm_stats,save_path)
    print('the accuarcy of model is:',np.mean(np.array(labels)==np.array(preds)))
        
    
if __name__=="__main__":
    
    model_path="model_parameter_Resnet_medicalnet\\32\\fold2_epoch0_val_0.0846_train_0.0851"
    device='cuda' if torch.cuda.is_available() else 'cpu'
    json_path="jsons\\patient_level\\train_val_split_new_dataset.json"
    excel_path="excel/vol_annotations_06_03_2025.xlsx"
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])
    save_excel_path="excel//misclassified_data.xlsx"
    batch_size=32
    num_workers=12
    timeout=600
    save_misclassified_data=False
    results_dir="results"
    data='test'#can either be train or test
    assert data=='train' or data=='test',"the only permitted values of data are train or test"
    os.makedirs(results_dir,exist_ok=True)
    save_file= model_path.split(os.sep)[0]+"_"+model_path.split(os.sep)[-2]+"_"+model_path.split(os.sep)[-1]+f'confusion_matrix_{data}_set_{json_path.split(os.sep)[-1].split(".")[0]}.png'
    print(save_file)

    evaluate_dataset(json_path,device,model_path,excel_path,transform,batch_size,num_workers,timeout,results_dir+os.sep+save_file,data,save_misclassified_data,save_excel_path)