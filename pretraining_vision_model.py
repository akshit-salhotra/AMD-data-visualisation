import argparse
import matplotlib
matplotlib.use('Agg')
from utils.logger import intialise_logger_nd_create_folders
from utils.util import load_model
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torch.nn import CrossEntropyLoss
from dataloader import OCTDataset,collate_fn
import torchvision.utils as vutils
import torch.optim as optim
from model.resnet_3d import Resnet18_3D
# from model.sequence_model import Seq_Model
# from model.resnet_medicalnet import resnet10
from model.auto_encoder import AutoEncoder
# from utils.make_plots import get_batch_stats_plot
from utils.util import SliceLevelPerceptualLoss
import os
from tqdm import tqdm
import logging
import json
# from sklearn.model_selection import KFold
import torch.nn as nn
import math
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# from hooks.batch_hook import create_hook,batchnorm_stats

def train_step(args,iter,data,label,epoch_loss,optimizer,model,criteron,dataset,num_folds):
    data=[d.to(args.device) for d in data]
    label=label.to(args.device)
    # print(image.shape)
    # with torch.no_grad():

    logits=model(*data)
        # print(logits.device,label.device)
    loss=criteron(logits,label)
    epoch_loss+=loss
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

        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument('--batch',type=float,default=4,help='batch size')
        parser.add_argument('--epoch',type=int,default=15,help='number of epoch')
        parser.add_argument('--json',type=str,default='jsons/train_test_val_split_without_scar_with_both_res.json',help="path of json file containing path of volumes")
        parser.add_argument('--excel-path',type=str,default='d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx',help='path of excel containing labels')
        parser.add_argument('--save-dir',type=str,default='model_parameter_Resnet_pretraining')
        parser.add_argument('--save-freq',type=int,default=5,help='after how many epochs are the parameters saved')
        parser.add_argument('--log-dir',type=str,default='logs/Resnet_pretraining',help='the directory in which training logs are to be saved')
        parser.add_argument('--gamma',type=float,default=0.1,help='gamma for learning rate decay')
        parser.add_argument('--step-size',type=int,default=10,help='number of epochs after which learning rate is to be decayed')
        parser.add_argument('--model-path',type=str,default=None,help='path of model parameters to be loaded')
        parser.add_argument('--device',type=torch.device,default=torch.device('cpu' if torch.cuda.is_available() else 'cpu'),help='computation device')
        parser.add_argument('--log-freq',type=int,default=5,help='after how many iterations losses are logged')
        parser.add_argument('--lambda_percep',type=float,default=0.8,help="weighting factor for perceptual loss")
        parser.add_argument('--model_ch',type=list,default=[16,32,64,128],help="channels in different layers of resnet")
        parser.add_argument('--save_recons',type=bool,default=True,help="whether to save some reconstructions")
        parser.add_argument('--save_bscans',type=torch.Tensor,default=torch.tensor([36,48,60,72]),help="which reconstructed bscans to save")
        parser.add_argument('--class_dict',type=dict,default={'early':0,'inter':1,'ga':2,'wet':3,'notAMD':4})

        args = parser.parse_args()
        
        param_dir=intialise_logger_nd_create_folders(args)
        
        model=AutoEncoder(args.model_ch).to(args.device)
        # model=nn.DataParallel(model)
        logging.info(f'found {torch.cuda.device_count()} gpus!')
        logging.info(model)

        percep_criteron=SliceLevelPerceptualLoss()
        recons_criteron=nn.MSELoss()
        optimizer=optim.Adam(model.parameters(),args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        
        if args.model_path:
            load_model(args,model)
            
        with open(args.json,'r') as file:
            paths= json.load(file) 
        train_paths=paths['train_path']
        val_paths=paths['test_path']
        transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.RandomHorizontalFlip(),
                                      transforms.Resize((256,256))])
        
        
        train_dataset=OCTDataset(train_paths,args.excel_path,transform,attn=False,undersample=True,classes=args.class_dict)
        val_dataset=OCTDataset(val_paths,args.excel_path,transform,attn=False,undersample=False,classes=args.class_dict)

        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,num_workers=0,timeout=0,collate_fn=collate_fn)
        test_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False,num_workers=0,timeout=0,collate_fn=collate_fn)
        
        
        
        if args.model_path and re.search(r'fold(\d+)',args.model_path):
            # if fold<int(re.search(r'fold(\d+)',args.model_path).group(1)):
                # continue
            # elif fold==int(re.search(r'fold(\d+)',args.model_path).group(1)):
                match=re.search(r'epoch(\d+)',args.model_path)
                start_epoch=int(match.group(1))+1
                pbar=tqdm(range(start_epoch,args.epoch))
        else:
            pbar=tqdm(range(args.epoch))

        for i in pbar:
                epoch_loss=0
                recons_loss=0
                percep_loss=0
                model.train()

                for iter,(data,_,_) in tqdm(enumerate(train_loader)):
                    
                    # print('hi...')
                    scans=data[0].to(args.device)
                    # print(torch.cuda.memory_summary())
                    recons_scans=model(scans)
                    # print('bye')
                    # print(torch.cuda.memory_summary())

                    r_loss=recons_criteron(recons_scans,scans)
                    p_loss=percep_criteron(recons_scans.reshape(-1,1,256,256),scans.reshape(-1,1,256,256))
                    # print('calculating loss')
                    loss=r_loss+args.lambda_percep*p_loss
                    # print(torch.cuda.memory_summary())
                    # print('....')
                    if iter% args.log_freq==0:
                        logging.info(f'Epoch:{i}/{args.epoch} iteration:{iter}/{math.ceil(len(train_dataset)/args.batch)} Loss is :{loss:.4f} reconstruction loss :{r_loss:.4f} perceptual loss :{p_loss:.4f}')

                    epoch_loss+=loss
                    recons_loss+=r_loss
                    percep_loss+=p_loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss/=len(train_dataset)
                percep_loss/=len(train_dataset)
                recons_loss/=len(train_dataset)

                logging.info(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f} reconstruction loss :{recons_loss:.4f} perceptual loss :{percep_loss:.4f}')
                pbar.set_postfix({'average epoch loss':f'{epoch_loss:.4f}'
                                  ,'average r loss':f'{recons_loss:.4f}'
                                  ,'average p loss':f'{percep_loss:.4f}'})
                scheduler.step()
                
                if i%args.save_freq==0 or args.epoch==i+1:
                    val_loss=0
                    val_recons_loss=0
                    val_percep_loss=0
                    model.eval()
                    
                    with torch.no_grad():
                        for i,(data,_,_) in enumerate(test_loader):
                            
                            scans=data[0].to(args.device)
                            recons_scans=model(scans)
                            val_r_loss=recons_criteron(recons_scans,scans)
                            val_p_loss=recons_criteron(recons_scans,scans)
                            loss=val_r_loss +args.lambda_percep*val_p_loss
                            
                            val_loss+=val_loss
                            val_recons_loss+=r_loss
                            val_percep_loss+=p_loss
                            
                        if args.save_recons:
                            recons_dir=param_dir+os.sep+"reconstructions"+os.sep+f"epoch_{i}"
                            os.makedirs(recons_dir,exist_ok=True)
                            b_scans=[]
                            
                            for index in range(min(4,args.batch)):
                                b_scans.append(recons_scans[index,0,args.save_bscans].unsqueeze(1))
                                b_scans.append(scans[index,0,args.save_bscans].unsqueeze(1))

                            vutils.save_image(b_scans,recons_dir+os.sep+str(i)+".png",normalize=True,nrow=args.save_bscans.shape[0])
                            
                        val_loss/=math.ceil(len(val_dataset)/args.batch)
                        val_recons_loss/=math.ceil(len(val_dataset)/args.batch)
                        val_percep_loss/=math.ceil(len(val_dataset)/args.batch)

                        logging.info(f'VAL loss :{val_loss:.4f} reconstruction loss :{val_recons_loss:.4f} perceptual loss :{val_percep_loss:.4f}')

                        torch.save(model.state_dict(),f'{param_dir}/epoch{i}_val_{val_loss:.4f}_train_{epoch_loss:.4f}')

            
        print('training completed !!')
        logging.info('training complete!!!')

    except Exception as e:
        print(e)
        logging.error(str(e))


