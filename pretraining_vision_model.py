import argparse
import matplotlib
matplotlib.use('Agg')
from utils.logger import intialise_logger_nd_create_folders
from utils.util import load_model
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset,random_split
# from torch.nn import CrossEntropyLoss
from memory_profiler import profile
from dataloader import OCTDataset,collate_fn,B_ScanDataset
import torchvision.utils as vutils
import torch.optim as optim
from model.resnet_3d import Resnet18_3D
# from model.sequence_model import Seq_Model
# from model.resnet_medicalnet import resnet10
from model.auto_encoder import AutoEncoder,AutoEncoder_2d
from utils.util import SliceLevelPerceptualLoss
import os
from tqdm import tqdm
import logging
import json
import torch.nn as nn
import math
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import wandb

def add_noise(x):
    return x + 0.001 * torch.randn_like(x)

def save_reconstructions_3d(args,recons_scans,scans,idx=None):
    b_scans=[]
    for index in range(min(4,args.batch)):
        b_scans.append(recons_scans[index,0,args.save_bscans].unsqueeze(1))
        b_scans.append(scans[index,0,args.save_bscans].unsqueeze(1))
    # print('bscan shape ',len(b_scans),b_scans[0].shape)
    b_scans=torch.concat(b_scans,dim=0)
    if idx is None:
        images = [wandb.Image(bscan, caption=f"Image {i}") for i, bscan in enumerate(b_scans)]
        wandb.log({"reconstructed bscans": images})
    else:
        vutils.save_image(b_scans,recons_dir+os.sep+"_"+str(idx)+".png",normalize=True,nrow=args.save_bscans.shape[0])

def save_reconstructions_2d(args,recons_scans,scans,idx=None):
    b_scans=[]
    # for index in range(min(8,args.batch)):
    b_scans.append(recons_scans[:min(8,args.batch)])
    b_scans.append(scans[:min(8,args.batch)])
    b_scans=torch.concat(b_scans,dim=0)

    if idx is None:
        images = [wandb.Image(bscan, caption=f"Image {i}") for i, bscan in enumerate(b_scans)]
        wandb.log({"reconstructions": images})

    else:
        vutils.save_image(b_scans,recons_dir+os.sep+"_"+str(idx)+".png",normalize=True,nrow=2)

    
    
if __name__=="__main__":
    try:
    
        parser = argparse.ArgumentParser(description="train arguments")

        parser.add_argument("--lr", type=float, default=0.0015, help="learning rate")
        parser.add_argument('--batch',type=float,default=64,help='batch size')
        parser.add_argument('--epoch',type=int,default=25,help='number of epoch')
        parser.add_argument('--json',type=str,default='jsons/train_test_val_split_without_scar.json',help="path of json file containing path of volumes")
        parser.add_argument('--excel-path',type=str,default='excel\\vol_annotations_06_03_2025.xlsx',help='path of excel containing labels')
        parser.add_argument('--save-dir',type=str,default='model_parameter_2DAutoEncoder')
        parser.add_argument('--save-freq',type=int,default=5,help='after how many epochs are the parameters saved')
        parser.add_argument('--log-dir',type=str,default='logs/2DAutoEncoder',help='the directory in which training logs are to be saved')
        parser.add_argument('--gamma',type=float,default=0.1,help='gamma for learning rate decay')
        parser.add_argument('--step-size',type=int,default=10,help='number of epochs after which learning rate is to be decayed')
        parser.add_argument('--model-path',type=str,default=None,help='path of model parameters to be loaded')
        parser.add_argument('--device',type=torch.device,default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),help='computation device')
        parser.add_argument('--log-freq',type=int,default=5,help='after how many iterations losses are logged')
        parser.add_argument('--lambda_percep',type=float,default=0.01,help="weighting factor for perceptual loss")
        parser.add_argument('--model_ch',type=list,default=[64,128,256,512],help="channels in different layers of resnet")
        parser.add_argument('--save_recons',type=bool,default=True,help="whether to save some reconstructions")
        parser.add_argument('--save_bscans',type=torch.Tensor,default=torch.tensor([36,48,60,72]),help="which reconstructed bscans to save")
        parser.add_argument('--class_dict',type=dict,default={'early':0,'inter':1,'ga':2,'wet':3,'notAMD':4})
        parser.add_argument('--is_2D',type=bool,default=True)
        parser.add_argument('--encoder_type',type=str,default='resnet34',help='which encoder to use')

        args = parser.parse_args()
        
        param_dir=intialise_logger_nd_create_folders(args)
        run=wandb.init(project='Auto-encoder AMD'
        ,config=vars(args))
        
        # model=AutoEncoder(args.model_ch).to(args.device)
        model=AutoEncoder_2d(encoder_type=args.encoder_type).to(args.device)
        model=nn.DataParallel(model)
        logging.info(f'found {torch.cuda.device_count()} gpus!')
        logging.info(model)

        percep_criteron=SliceLevelPerceptualLoss()
        recons_criteron=nn.MSELoss()
        optimizer=optim.Adam(model.parameters(),args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        
        if args.model_path:
            load_model(args,model)
            
        if not args.is_2D:
            
            with open(args.json,'r') as file:
                paths= json.load(file) 
            train_paths=paths['train_path']
            val_paths=paths['test_path']
        
        transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.RandomHorizontalFlip(),
                                    transforms.Resize((256,256)),
                                    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                                    # transforms.Lambda(add_noise)
                                    ])
        
        if not args.is_2D:
            dataset_config={'transforms':transform,
                            'attn':False,
                            'undersample':True,
                            'denoise':False,
                            'classes':args.class_dict}
            
            train_dataset=OCTDataset(val_paths,args.excel_path,**dataset_config)
            val_dataset=OCTDataset(val_paths,args.excel_path,**dataset_config)
        # train_dataset = Subset(train_dataset, indices=range(4))
        else:
            denoise=False
            collate_fn=None
            dataset_config={'transform':transform,
                            'denoise':denoise}
            dataset=B_ScanDataset(transform,args.excel_path,denoise)
            val_percent = 0.02
            val_size = int(len(dataset) * val_percent)
            train_size = len(dataset) - val_size

            generator = torch.Generator().manual_seed(42)

            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
            # train_dataset=Subset(dataset,range(10))
            # val_dataset=Subset(dataset,range(10))
            print('number of scans in train set :',len(train_dataset))
            print('number of scans in val set',len(val_dataset))

        # run.config.update(dataset_config)
        # print('hi',collate_fn)
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,num_workers=15,timeout=300,collate_fn=collate_fn)
        test_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False,num_workers=15,timeout=300,collate_fn=collate_fn)
        
        
        
        if args.model_path and re.search(r'fold(\d+)',args.model_path):

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

                def train(args,data,recons_loss,percep_loss,epoch_loss):

                        if not args.is_2D:
                            scans=data[0].to(args.device)
                        else:
                            scans=data.to(args.device)
                            
                        recons_scans=model(scans)

                        r_loss=recons_criteron(recons_scans,scans)
                        if not args.is_2D:
                            assert recons_scans.shape[-2]==(256,256),f' shape is not right , found the shape to be :{recons_scans.shape}'
                            p_loss=percep_criteron(recons_scans.reshape(-1,1,256,256),scans.reshape(-1,1,256,256))
                        else:
                            p_loss=percep_criteron(recons_scans,scans)

                        # print('calculating loss')
                        loss=r_loss+args.lambda_percep*p_loss
                        # print(torch.cuda.memory_summary())
                        # print('....')
                        if iter% args.log_freq==0:
                            logging.info(f'Epoch:{i}/{args.epoch} iteration:{iter}/{math.ceil(len(train_dataset)/args.batch)} Loss is :{loss:.4f} reconstruction loss :{r_loss:.4f} perceptual loss :{p_loss:.4f}')
                        
                        if iter%(args.log_freq*5)==0:
                            if args.is_2D:
                                save_reconstructions_2d(args,recons_scans,scans)
                            else:
                                save_reconstructions_3d(args,recons_scans,scans)
                                
                        epoch_loss+=loss
                        recons_loss+=r_loss
                        percep_loss+=p_loss
        
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        return epoch_loss,percep_loss,recons_loss

                for iter,data in tqdm(enumerate(train_loader)):

                        epoch_loss,percep_loss,recons_loss=train(args,data,recons_loss,percep_loss,epoch_loss)


                epoch_loss/=len(train_dataset)
                percep_loss/=len(train_dataset)
                recons_loss/=len(train_dataset)
                train_metrics={'average epoch loss':f'{epoch_loss:.4f}'
                                  ,'average r loss':f'{recons_loss:.4f}'
                                  ,'average p loss':f'{percep_loss:.4f}'}
                
                logging.info(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f} reconstruction loss :{recons_loss:.4f} perceptual loss :{percep_loss:.4f}')
                pbar.set_postfix(train_metrics)
                wandb.log(train_metrics)
                scheduler.step()
                
                if i%args.save_freq==0 or args.epoch==i+1:
                    val_loss=0
                    val_recons_loss=0
                    val_percep_loss=0
                    model.eval()
                    
                    with torch.no_grad():
                        for i,data in enumerate(test_loader):
                            if not args.is_2D:
                                scans=data[0].to(args.device)
                            else:
                                scans=data.to(args.device)
                                
                            recons_scans=model(scans)
                            val_r_loss=recons_criteron(recons_scans,scans)
                            
                            if args.is_2D:
                                val_p_loss=percep_criteron(recons_scans,scans)
                            else:
                                val_p_loss=percep_criteron(recons_scans.reshape(-1,1,256,256),scans.reshape(-1,1,256,256))

                                
                            loss=val_r_loss +args.lambda_percep*val_p_loss
                            
                            val_loss+=val_loss
                            val_recons_loss+=val_r_loss
                            val_percep_loss+=val_p_loss
                            
                        if args.save_recons:
                            recons_dir=param_dir+os.sep+"reconstructions"+os.sep+f"epoch_{i}"
                            os.makedirs(recons_dir,exist_ok=True)
                            # b_scans=[]
                            
                            if not args.is_2D:
                                save_reconstructions_3d(args,recons_scans,scans,i)
                            else:
                                save_reconstructions_2d(args,recons_scans,scans,i)
                                # for index in range(min(4,args.batch)):
                                #     b_scans.append(recons_scans[index,0,args.save_bscans].unsqueeze(1))
                                #     b_scans.append(scans[index,0,args.save_bscans].unsqueeze(1))

                                # vutils.save_image(b_scans,recons_dir+os.sep+str(i)+".png",normalize=True,nrow=args.save_bscans.shape[0])
                            
                        val_loss/=math.ceil(len(val_dataset)/args.batch)
                        val_recons_loss/=math.ceil(len(val_dataset)/args.batch)
                        val_percep_loss/=math.ceil(len(val_dataset)/args.batch)

                        logging.info(f'VAL loss :{val_loss:.4f} reconstruction loss :{val_recons_loss:.4f} perceptual loss :{val_percep_loss:.4f}')
                        wandb.log({
                            'val loss':val_loss,
                            'val recons loss':val_recons_loss,
                            'val perceptual loss':val_percep_loss
                        })
                        torch.save(model.state_dict(),f'{param_dir}/epoch{i}_val_{val_loss:.4f}_train_{epoch_loss:.4f}')

            
        print('training completed !!')
        logging.info('training complete!!!')
        wandb.finish()

    except Exception as e:
        print(e)
        logging.error(str(e))


