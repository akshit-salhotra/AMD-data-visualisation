import argparse
import matplotlib
matplotlib.use('Agg')
from utils.logger import intialise_logger_nd_create_folders
from utils.util import load_model
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from memory_profiler import profile
from dataloader import OCTDataset,collate_fn
import torchvision.utils as vutils
import torch.optim as optim
from model.auto_encoder import AutoEncoder
from utils.util import SliceLevelPerceptualLoss
import os
from tqdm import tqdm
import logging
import json
import torch.nn as nn
import math
import re
import numpy as np

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0:
        torch.cuda.set_device(0)
    elif rank == 1:
        torch.cuda.set_device(2)

def cleanup():
    dist.destroy_process_group()

def get_devices(rank):
    # (model_device, perceptual_loss_device)
    return (torch.device(f"cuda:{0 if rank == 0 else 2}"),
            torch.device(f"cuda:{1 if rank == 0 else 3}"))

def train(rank, world_size,args):
    
    setup(rank, world_size)
    model_device, perceptual_device = get_devices(rank)

    with open(args.json,'r') as file:
            paths= json.load(file) 
            
    train_paths=paths['train_path']
    val_paths=paths['test_path']
    
    transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.RandomHorizontalFlip(),
                                      transforms.Resize((256,256))])
    # Dataset and DataLoader
    train_dataset=OCTDataset(train_paths,args.excel_path,transform,attn=False,undersample=True,classes=args.class_dict)
    val_dataset=OCTDataset(val_paths,args.excel_path,transform,attn=False,undersample=False,classes=args.class_dict)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler=DistributedSampler(val_dataset,num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset
                            , batch_size=args.batch 
                            ,shuffle=True
                            ,num_workers=0
                            ,timeout=0
                            ,collate_fn=collate_fn
                            , sampler=train_sampler
                            , pin_memory=True)
    
    val_dataloader = DataLoader(val_dataset
                            , batch_size=args.batch 
                            ,shuffle=True
                            ,num_workers=0
                            ,timeout=0
                            ,collate_fn=collate_fn
                            , sampler=val_sampler
                            , pin_memory=True)

    # Model and loss
    model = AutoEncoder.to(model_device)
    if rank==0:
        logging.info(model)
        
    model = DDP(model, device_ids=[model_device.index])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    recons_criteron = nn.MSELoss().to(model_device)
    perceptual_loss_fn =SliceLevelPerceptualLoss().to(perceptual_device)

    if args.model_path:
            load_model(args,model)
    
    if args.model_path and re.search(r'fold(\d+)',args.model_path):
  
        match=re.search(r'epoch(\d+)',args.model_path)
        start_epoch=int(match.group(1))+1
        pbar=range(start_epoch,args.epoch)
    else:
        pbar=range(args.epoch)
            
    for epoch in pbar:
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        for data,_,_ in train_dataloader:
            scans=data[0].to(model_device)
            recons_scans=model(scans)
            r_loss=recons_criteron(recons_scans,scans)
            
            with torch.cuda.device(perceptual_device):
                recons_scans = recons_scans.to(perceptual_device, non_blocking=True)
                scans = scans.to(perceptual_device, non_blocking=True)
                p_loss = perceptual_loss_fn(recons_scans.reshape(-1,1,256,256), scans.reshape(-1,1,256,256))

            loss=r_loss+args.lambda_percep*p_loss.to(model_device)

            if iter% args.log_freq==0:
                logging.info(f'Rank: {rank} Epoch:{epoch}/{args.epoch} iteration:{iter}/{math.ceil(len(train_dataset)/args.batch)} Loss is :{loss:.4f} reconstruction loss :{r_loss:.4f} perceptual loss :{p_loss:.4f}')

            epoch_loss+=loss
            recons_loss+=r_loss
            percep_loss+=p_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss/=math.ceil(len(train_dataset/world_size)/args.batch)
        percep_loss/=math.ceil(len(train_dataset/world_size)/args.batch)
        recons_loss/=math.ceil(len(train_dataset/world_size)/args.batch)
        
        print(f"[Rank {rank}] Epoch {epoch} completed.")
        scheduler.step()
        
        if epoch%args.save_freq==0 or args.epoch==epoch+1:
            val_loss=0
            val_recons_loss=0
            val_percep_loss=0
            model.eval()
            
            with torch.no_grad():
                for idx,(data,_,_) in enumerate(val_dataloader):
                    
                    scans=data[0].to(model_device)
                    recons_scans=model(scans)
                    val_r_loss=recons_criteron(recons_scans,scans)
                    
                    with torch.cuda.device(perceptual_device):
                        recons_scans = recons_scans.to(perceptual_device, non_blocking=True)
                        scans = scans.to(perceptual_device, non_blocking=True)
                        val_p_loss = perceptual_loss_fn(recons_scans.reshape(-1,1,256,256), scans.reshape(-1,1,256,256))

                    loss=val_r_loss +args.lambda_percep*val_p_loss.to(model_device)
                    
                    val_loss+=loss
                    val_recons_loss+=val_r_loss
                    val_percep_loss+=val_p_loss
                    
                if args.save_recons:
                    recons_dir=param_dir+os.sep+"reconstructions"+os.sep+f"epoch_{i}"
                    os.makedirs(recons_dir,exist_ok=True)
                    b_scans=[]
                    
                    for index in range(min(4,args.batch)):
                        b_scans.append(recons_scans[index,0,args.save_bscans].unsqueeze(1))
                        b_scans.append(scans[index,0,args.save_bscans].unsqueeze(1))

                    vutils.save_image(b_scans,recons_dir+os.sep+'rank_'+str(rank)+"_"+str(idx)+".png",normalize=True,nrow=args.save_bscans.shape[0])
                    
                val_loss/=math.ceil(len(val_dataset/world_size)/args.batch)
                val_recons_loss/=math.ceil(len(val_dataset/world_size)/args.batch)
                val_percep_loss/=math.ceil(len(val_dataset/world_size)/args.batch)

                logging.info(f'Rank: {rank} VAL loss :{val_loss:.4f} reconstruction loss :{val_recons_loss:.4f} perceptual loss :{val_percep_loss:.4f}')
                
                if rank==0:
                    torch.save(model.state_dict(),f'{param_dir}/epoch{epoch}_val_{val_loss:.4f}_train_{epoch_loss:.4f}')

    cleanup()

# if __name__=="__main__":
try:
    
        parser = argparse.ArgumentParser(description="train arguments")

        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument('--batch',type=float,default=1,help='batch size')
        parser.add_argument('--epoch',type=int,default=2,help='number of epoch')
        parser.add_argument('--json',type=str,default='jsons/train_test_val_split_without_scar.json',help="path of json file containing path of volumes")
        parser.add_argument('--excel-path',type=str,default='d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx',help='path of excel containing labels')
        parser.add_argument('--save-dir',type=str,default='model_parameter_Resnet_pretraining')
        parser.add_argument('--save-freq',type=int,default=5,help='after how many epochs are the parameters saved')
        parser.add_argument('--log-dir',type=str,default='logs/Resnet_pretraining',help='the directory in which training logs are to be saved')
        parser.add_argument('--gamma',type=float,default=0.1,help='gamma for learning rate decay')
        parser.add_argument('--step-size',type=int,default=10,help='number of epochs after which learning rate is to be decayed')
        parser.add_argument('--model-path',type=str,default=None,help='path of model parameters to be loaded')
        parser.add_argument('--device',type=torch.device,default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),help='computation device')
        parser.add_argument('--log-freq',type=int,default=5,help='after how many iterations losses are logged')
        parser.add_argument('--lambda_percep',type=float,default=0.8,help="weighting factor for perceptual loss")
        parser.add_argument('--model_ch',type=list,default=[64,128,256,512],help="channels in different layers of resnet")
        parser.add_argument('--save_recons',type=bool,default=True,help="whether to save some reconstructions")
        parser.add_argument('--save_bscans',type=torch.Tensor,default=torch.tensor([36,48,60,72]),help="which reconstructed bscans to save")
        parser.add_argument('--class_dict',type=dict,default={'early':0,'inter':1,'ga':2,'wet':3,'notAMD':4})

        args = parser.parse_args()
        
        param_dir=intialise_logger_nd_create_folders(args)
        
        # model=AutoEncoder(args.model_ch).to(args.device)
        # model=nn.DataParallel(model)
        logging.info(f'found {torch.cuda.device_count()} gpus!')
        # logging.info(model)
        train()
        # percep_criteron=SliceLevelPerceptualLoss()
        # recons_criteron=nn.MSELoss()
        # optimizer=optim.Adam(model.parameters(),args.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        
        # if args.model_path:
        #     load_model(args,model)
            
        # with open(args.json,'r') as file:
        #     paths= json.load(file) 
        # train_paths=paths['train_path']
        # val_paths=paths['test_path']
        # transform=transforms.Compose([transforms.ToTensor(),
        #                             #   transforms.RandomHorizontalFlip(),
        #                               transforms.Resize((256,256))])
        
        # train_dataset=OCTDataset(val_paths,args.excel_path,transform,attn=False,undersample=True,classes=args.class_dict)
        # val_dataset=OCTDataset(val_paths,args.excel_path,transform,attn=False,undersample=False,classes=args.class_dict)
        # train_dataset = Subset(train_dataset, indices=range(4))

        # train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,num_workers=0,timeout=0,collate_fn=collate_fn)
        # test_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False,num_workers=0,timeout=0,collate_fn=collate_fn)
        
        
        
        # if args.model_path and re.search(r'fold(\d+)',args.model_path):
        #     # if fold<int(re.search(r'fold(\d+)',args.model_path).group(1)):
        #         # continue
        #     # elif fold==int(re.search(r'fold(\d+)',args.model_path).group(1)):
        #         match=re.search(r'epoch(\d+)',args.model_path)
        #         start_epoch=int(match.group(1))+1
        #         pbar=tqdm(range(start_epoch,args.epoch))
        # else:
        #     pbar=tqdm(range(args.epoch))

#         for i in pbar:
#                 epoch_loss=0
#                 recons_loss=0
#                 percep_loss=0
#                 model.train()

#                 def train(data,recons_loss,percep_loss,epoch_loss):
                    
#                         # print('hi...')
#                         scans=data[0].to(args.device)
#                         # print(torch.cuda.memory_summary())
#                         recons_scans=model(scans)
#                         # print('bye')
#                         # print(torch.cuda.memory_summary())

#                         r_loss=recons_criteron(recons_scans,scans)
#                         p_loss=0
#                         # p_loss=percep_criteron(recons_scans.reshape(-1,1,256,256),scans.reshape(-1,1,256,256))
#                         # print('calculating loss')
#                         loss=r_loss+args.lambda_percep*p_loss
#                         # print(torch.cuda.memory_summary())
#                         # print('....')
#                         if iter% args.log_freq==0:
#                             logging.info(f'Epoch:{i}/{args.epoch} iteration:{iter}/{math.ceil(len(train_dataset)/args.batch)} Loss is :{loss:.4f} reconstruction loss :{r_loss:.4f} perceptual loss :{p_loss:.4f}')

#                         epoch_loss+=loss
#                         recons_loss+=r_loss
#                         percep_loss+=p_loss
        
#                         loss.backward()
#                         optimizer.step()
#                         optimizer.zero_grad()
#                         return epoch_loss,percep_loss,recons_loss
                    
# #                 with torch.profiler.profile(schedule=torch.profiler.schedule(
# #         wait=1,        # don't record the first step
# #         warmup=1,      # record shapes and memory after warmup
# #         active=2       # record 2 steps of real data
# #     ),
# #     activities=[
# #         torch.profiler.ProfilerActivity.CPU,
# #         torch.profiler.ProfilerActivity.CUDA],
# #     profile_memory=True,
# #     record_shapes=True,
# #     with_stack=True
# # ) as prof:
#                 for iter,(data,_,_) in tqdm(enumerate(train_loader)):

#                         epoch_loss,percep_loss,recons_loss=train(data,recons_loss,percep_loss,epoch_loss)
#                         # prof.step()
                        
#                     # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

#                 epoch_loss/=len(train_dataset)
#                 percep_loss/=len(train_dataset)
#                 recons_loss/=len(train_dataset)

#                 logging.info(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f} reconstruction loss :{recons_loss:.4f} perceptual loss :{percep_loss:.4f}')
#                 pbar.set_postfix({'average epoch loss':f'{epoch_loss:.4f}'
#                                   ,'average r loss':f'{recons_loss:.4f}'
#                                   ,'average p loss':f'{percep_loss:.4f}'})
#                 scheduler.step()
                
#                 if i%args.save_freq==0 or args.epoch==i+1:
#                     val_loss=0
#                     val_recons_loss=0
#                     val_percep_loss=0
#                     model.eval()
                    
#                     with torch.no_grad():
#                         for i,(data,_,_) in enumerate(test_loader):
                            
#                             scans=data[0].to(args.device)
#                             recons_scans=model(scans)
#                             val_r_loss=recons_criteron(recons_scans,scans)
#                             # val_p_loss=percep_criteron(recons_scans,scans)
#                             val_p_loss=0
#                             loss=val_r_loss +args.lambda_percep*val_p_loss
                            
#                             val_loss+=val_loss
#                             val_recons_loss+=val_r_loss
#                             val_percep_loss+=val_p_loss
                            
#                         if args.save_recons:
#                             recons_dir=param_dir+os.sep+"reconstructions"+os.sep+f"epoch_{i}"
#                             os.makedirs(recons_dir,exist_ok=True)
#                             b_scans=[]
                            
#                             for index in range(min(4,args.batch)):
#                                 b_scans.append(recons_scans[index,0,args.save_bscans].unsqueeze(1))
#                                 b_scans.append(scans[index,0,args.save_bscans].unsqueeze(1))

#                             vutils.save_image(b_scans,recons_dir+os.sep+str(i)+".png",normalize=True,nrow=args.save_bscans.shape[0])
                            
#                         val_loss/=math.ceil(len(val_dataset)/args.batch)
#                         val_recons_loss/=math.ceil(len(val_dataset)/args.batch)
#                         val_percep_loss/=math.ceil(len(val_dataset)/args.batch)

#                         logging.info(f'VAL loss :{val_loss:.4f} reconstruction loss :{val_recons_loss:.4f} perceptual loss :{val_percep_loss:.4f}')

#                         torch.save(model.state_dict(),f'{param_dir}/epoch{i}_val_{val_loss:.4f}_train_{epoch_loss:.4f}')

            
#         print('training completed !!')
#         logging.info('training complete!!!')

except Exception as e:
        print(e)
        logging.error(str(e))


