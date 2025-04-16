import argparse
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset
from torch.nn import CrossEntropyLoss
from dataloader import OCTDataset,collate_fn
import torch.optim as optim
from model.resnet_3d import Resnet18_3D
# from model.sequence_model import Seq_Model
import os
from tqdm import tqdm
import logging
import json
from sklearn.model_selection import KFold
import torch.nn as nn
import sys
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def intialise_logger_nd_create_folders(args):
    # if args.model_path:
        # logging.info('resumed training')
        # num_folder=args.model_path.split(os.sep)[1]
        # print(num_folder)
        # save_path=args.save_dir+os.sep+num_folder
    # else:
    if os.path.exists(args.save_dir):
            num_folder=str(int(sorted(os.listdir(args.save_dir),key=lambda x:int(x))[-1])+1)
    else:
            os.makedirs(args.save_dir,exist_ok=False)
            num_folder='0'
    save_path=args.save_dir+os.sep+num_folder
    os.makedirs(save_path)


    os.makedirs(args.log_dir,exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,                  
        format='%(asctime)s - %(levelname)s - %(message)s',  
        datefmt='%Y-%m-%d %H:%M:%S',        
        handlers=[
            # logging.StreamHandler(),        
            logging.FileHandler(args.log_dir+os.sep+f"train_{num_folder}.log")  
        ]
    )
    print('logging at:',args.log_dir+os.sep+f"train_{num_folder}.log")
    logging.info('------------------------------------------------------------------------------------------------')
    logging.info('starting new training !!!!')
    logging.info('------------------------------------------------------------------------------------------------')

    logging.info(args)

        
    logging.info(f'parameters are being saved at :{save_path}')
    
    return save_path

def load_model(args):
    model.load_state_dict(torch.load(args.model_path))
    print('loaded model parameters from ',args.model_path)
    logging.info(f'loaded model parameters from {args.model_path}')

def train_step(args,iter,data,label,epoch_loss,optimizer,model,criteron,dataset,num_folds):
    data=[d.to(args.device) for d in data]
    label=label.to(args.device)
    # print(image.shape)
    logits=model(*data)
    # print(logits.device,label.device)
    loss=criteron(logits,label)
    epoch_loss+=loss
    with torch.no_grad():
        if iter%5==0:
            # print(logits,label)
            logging.info(f'epoch:{i}/{args.epoch} iteration:{iter}/{(len(dataset)*(num_folds-1))//(num_folds*args.batch)+1} batch loss is :{loss:.4f}')
            logging.info(f'the acc is :{torch.mean((torch.argmax(logits,dim=-1)==label).to(torch.float32)).detach().cpu()}')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return epoch_loss,torch.argmax(logits,dim=-1)

def val_step(args,data,label,val_loss,model,criteron):
    data=[d.to(args.device) for d in data]
    label=label.to(args.device)
    logits=model(*data)
    loss=criteron(logits,label)
    val_loss+=loss
    preds=torch.argmax(logits,dim=-1)
    assert label.shape==preds.shape, 'the number of labels and images do not match'
    return val_loss,preds,label


if __name__=="__main__":
    try:
    
        parser = argparse.ArgumentParser(description="train arguments")

        parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
        parser.add_argument('--batch',type=float,default=16,help='batch size')
        parser.add_argument('--epoch',type=int,default=25,help='number of epoch')
        parser.add_argument('--json',type=str,default='jsons/train_test_val_split_without_scar.json',help="path of json file containing path of volumes")
        parser.add_argument('--excel-path',type=str,default='d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx',help='path of excel containing labels')
        parser.add_argument('--save-dir',type=str,default='model_parameter_Resnet')
        parser.add_argument('--save-freq',type=int,default=2,help='after how many epochs are the parameters saved')
        parser.add_argument('--log-dir',type=str,default='logs/Resnet',help='the directory in which training logs are to be saved')
        parser.add_argument('--gamma',type=float,default=0.1,help='gamma for learning rate decay')
        parser.add_argument('--step-size',type=int,default=10,help='number of epochs after which learning rate is to be decayed')
        parser.add_argument('--model-path',type=str,default=None,help='path of model parameters to be loaded')
        parser.add_argument('--device',type=torch.device,default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),help='computation device')
        parser.add_argument('--weight_matrix',type=torch.tensor,default=torch.tensor([0.25,0.25,0.125,0.16,1]),help='weights for weighted cross entropy')
        parser.add_argument('--class_dict',type=dict,default={'early':0,'inter':1,'ga':2,'wet':3,'notAMD':4})
        parser.add_argument('--num_classes',type=int,default=5,help="number of classes of the classifier")

        args = parser.parse_args()
        
        param_dir=intialise_logger_nd_create_folders(args)
        
        model=Resnet18_3D(num_classes=args.num_classes).to(args.device)
        # model=Seq_Model(num_classes=args.num_classes,device=args.device).to(args.device)
        model=nn.DataParallel(model)
        logging.info(f'found {torch.cuda.device_count()} gpus!')
        logging.info(model)
        criteron=CrossEntropyLoss(weight=args.weight_matrix.to(args.device))##make sure to add weight factor
        #early,inter,ga,wet,scar,notamd
        optimizer=optim.Adam(model.parameters(),args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        
        if args.model_path:
            load_model(args)
            
        with open(args.json,'r') as file:
            paths= json.load(file) 
        train_paths=paths['train_path']

        transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.RandomHorizontalFlip(),
                                      transforms.Resize((256,256))])
        dataset=OCTDataset(train_paths,args.excel_path,transform,attn=False,undersample=False,classes=args.class_dict)
        num_folds=2
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        # for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        for fold in range(1):

            logging.info(f'fold number:{str(fold)}')
            print(f'fold number :',fold)

            train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,num_workers=8,timeout=600,collate_fn=collate_fn)
            # test_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch, shuffle=False,num_workers=8,timeout=600,collate_fn=collate_fn)
            if args.model_path:
                if fold<int(re.search(r'fold(\d+)',args.model_path).group(1)):
                    continue
                elif fold==int(re.search(r'fold(\d+)',args.model_path).group(1)):
                    match=re.search(r'epoch(\d+)',args.model_path)
                    start_epoch=int(match.group(1))+1
                    pbar=tqdm(range(start_epoch,args.epoch))
                else:
                     pbar=tqdm(range(args.epoch))
            else:
                pbar=tqdm(range(args.epoch))

            for i in pbar:
                    epoch_loss=0
                    model.train()
                    preds=[]
                    labels=[]
                    for iter,(data,label) in tqdm(enumerate(train_loader)):
                        # print(image.shape)
                        # print(label)
                        epoch_loss,pred=train_step(args,iter,data,label,epoch_loss,optimizer,model,criteron,dataset,num_folds)
                        preds.extend(pred.detach().cpu().tolist())
                        labels.extend(label.detach().cpu().tolist())
                        
                    epoch_loss/=((len(dataset)*(num_folds-1))//(num_folds*args.batch)+1)
                    cm=confusion_matrix(labels,preds)
                    classes=["early","inter","ga","wet","notamd"]

                    plt.figure(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=classes,yticklabels=classes)
                    plt.title(f'Confusion Matrix,acc:{np.mean(np.array(labels)==np.array(preds)):.4f}')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')

                    # Save as image
                    plt.savefig(f"{param_dir+os.sep}confusion_matrix_fold_{fold}_epoch_{i}", dpi=300)
                    logging.info(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f}')
                    pbar.set_postfix({'average epoch loss':f'{epoch_loss:.4f}'})
                    scheduler.step()
                    
                    if i%args.save_freq==0 or args.epoch==i+1:
    #                     val_loss=0
    #                     correct_pred=0
    #                     model.eval()
    #                     val_labels=[]
    #                     val_preds=[]
    #                     with torch.no_grad():
    #                         for (data,label) in test_loader:
    #                             # print(data,label)
    #                             val_loss,preds,label=val_step(args,data,label,val_loss,model,criteron)
    #                             # print(preds.device,label.device)
    #                             correct_pred+=(label==preds).sum().item()
    #                             val_labels.extend(label.detach().cpu().tolist())
    #                             val_preds.extend(preds.detach().cpu().tolist())

    #                         logging.info(f'val accuarcy is : {correct_pred/(len(dataset)/num_folds):.4f}')
    #                         val_loss/=(len(dataset)//(num_folds*args.batch)+1)
    #                         logging.info(f'VAL loss :{val_loss:.4f}')
    #                         cm=confusion_matrix(val_labels,val_preds)
    #                         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=classes,yticklabels=classes)
    #                         plt.title(f'Confusion Matrix,acc:{np.mean(np.array(labels)==np.array(preds)):.4f}')
    #                         plt.xlabel('Predicted')
    #                         plt.ylabel('Actual')
    #                         plt.savefig(f'{param_dir+os.sep}confusion_matrix_val_fold_{fold}_epoch_{i}.png')
                            torch.save(model.state_dict(),f'{param_dir}/fold{fold}_epoch{i}_val_{epoch_loss:.4f}_train_{epoch_loss:.4f}')
                
    #     print('training completed !!')
    #     logging.info('training complete!!!')
    
    except Exception as e:
        print(e)
        logging.error(str(e))
    