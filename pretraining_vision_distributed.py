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
from utils.util import SliceLevelPerceptualLoss,PixelWiseWeightedMSE
import os
from tqdm import tqdm
import logging
import json
import torch.nn as nn
import math
import re
import numpy as np
import torch.multiprocessing as mp
import wandb

def intialise_logger(log_dir,rank):
    logging.basicConfig(
            level=logging.INFO,                  
            format='%(asctime)s - %(levelname)s - %(message)s',  
            datefmt='%Y-%m-%d %H:%M:%S',        
            handlers=[
                # logging.StreamHandler(),        
                logging.FileHandler(log_dir+os.sep+f"train_{rank}.log")  
            ]
        )
    print('logging at:',log_dir+os.sep+f"train_{rank}.log")
       
        # logging.info(args)

            
        # logging.info()

def save_recons(param_dir,epoch,recons_scans,scans,rank,idx,args):
    recons_dir=param_dir+os.sep+"reconstructions"+os.sep+f"epoch_{epoch}"
    os.makedirs(recons_dir,exist_ok=True)
    b_scans=[]
    
    for index in range(min(4,args.batch)):
        b_scans.append(recons_scans[index,0,args.save_bscans].unsqueeze(1))
        b_scans.append(scans[index,0,args.save_bscans].unsqueeze(1))
    # print('bscan shape ',len(b_scans),b_scans[0].shape)
    b_scans=torch.concat(b_scans,dim=0)
    vutils.save_image(b_scans,recons_dir+os.sep+'rank_'+str(rank)+"_"+str(idx)+".png",normalize=True,nrow=args.save_bscans.shape[0])
                        
def log_train_config(args,param_dir):
    logging.info(args)
    logging.info(f'parameters are being saved at :{param_dir}')
    model=AutoEncoder()
    logging.info(model)
    del model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.set_flush_denormal(True)
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
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

def train(rank, world_size,args,param_dir,log_dir):
    
    if rank==0:
       run = wandb.init(
    project='Auto-encoder AMD',
    config=vars(args))
       run.config.update({'arch':'3d resnet'
                              })
    setup(rank, world_size)
    print('starting process:',rank)
    intialise_logger(log_dir,rank)
    model_device, perceptual_device = get_devices(rank)

    with open(args.json,'r') as file:
            paths= json.load(file) 
            
    train_paths=paths['train_path']
    val_paths=paths['test_path']
    
    transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.RandomHorizontalFlip(),
                                      transforms.Resize((256,256))])
    # Dataset and DataLoader
    train_dataloader_config={'attn':False
                        ,'undersample':True
                        ,'denoise':True
                        ,'classes':args.class_dict
                        ,'multiThread':True
                        ,'transforms':transform}
    val_dataloader_config=train_dataloader_config

    if rank==0:
        run.config.update(train_dataloader_config)
        run.config.update(val_dataloader_config)
        
    train_dataset=OCTDataset(train_paths,args.excel_path,**train_dataloader_config)
    val_dataset=OCTDataset(val_paths,args.excel_path,**val_dataloader_config)
    # val_dataset=Subset(val_dataset,range(1))
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler=DistributedSampler(val_dataset,num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset
                            , batch_size=args.batch 
                            ,num_workers=0
                            ,timeout=0
                            ,collate_fn=collate_fn
                            , sampler=train_sampler
                            , pin_memory=True)
    
    val_dataloader = DataLoader(val_dataset
                            , batch_size=args.batch
                            ,num_workers=0
                            ,timeout=0
                            ,collate_fn=collate_fn
                            , sampler=val_sampler
                            , pin_memory=True
                        )

    # Model and loss
    model = AutoEncoder()
    model.to(model_device)

    # if rank==0:
    #     logging.info(model)
        
    model = DDP(model, device_ids=[model_device.index],find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if rank==0:
        wandb.watch(model.module,log='all',log_freq=args.log_freq)
    recons_criteron = nn.MSELoss().to(model_device)
    # recons_criteron=PixelWiseWeightedMSE(scaling_function=args.scaling_fn).to(model_device)
    recons_config={}
    if args.scaling_fn=='adaptiveHybridSigmoid':
        recons_config={
            'intial_k':0
            ,'theta':0.4
            ,'delta':0.01
            ,'rank':rank
            }
    if  rank==0:
        run.config.update(recons_config)   
    perceptual_loss_fn =SliceLevelPerceptualLoss(layer=args.percep_layer).to(perceptual_device)
    # perceptual_loss_fn=nn.MSELoss()

    if args.model_path:
            load_model(args,model)
    
    if args.model_path and re.search(r'fold(\d+)',args.model_path):
  
        match=re.search(r'epoch(\d+)',args.model_path)
        start_epoch=int(match.group(1))+1
        pbar=tqdm(range(start_epoch,args.epoch))
    else:
        pbar=tqdm(range(args.epoch))
            
    for epoch in pbar:
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        epoch_loss=0
        percep_loss=0
        recons_loss=0

        for iteration,(data,_,_) in enumerate(train_dataloader):
            scans=data[0].to(model_device)
            recons_scans=model(scans)
            if args.scaling_fn=='adaptiveHybridSigmoid':
                recons_config['iteration']=iteration
            r_loss=recons_criteron(recons_scans,scans,**recons_config)
            
            p_loss=0
            with torch.cuda.device(perceptual_device):
                recons_scans = recons_scans.to(perceptual_device, non_blocking=True)
                scans = scans.to(perceptual_device, non_blocking=True)
                # print(scans.device)
                # print(scans.shape)
                p_loss = perceptual_loss_fn(recons_scans.reshape(-1,1,256,256)[::args.bscan_step], scans.reshape(-1,1,256,256)[::args.bscan_step])
                # print(torch.cuda.memory_summary())
            loss=r_loss+args.lambda_percep*p_loss.to(model_device)
            # loss=r_loss

            gathered_losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, loss)
            gathered_r_losses = [torch.zeros_like(r_loss) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_r_losses, r_loss)
            gathered_p_losses = [torch.zeros_like(p_loss) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_p_losses, p_loss)

            if dist.get_rank() == 0:
                for i, (l,p,r) in enumerate(zip(gathered_losses,gathered_p_losses,gathered_r_losses)):
                    wandb.log({f"loss_rank_{i}": l.item()
                               ,f'reconstruction loss rank {i}':r.item()
                               ,f'perceptual loss rank {i}':p.item()})

            if iteration% args.log_freq==0:
                logging.info(f'Rank: {rank} Epoch:{epoch}/{args.epoch} iteration:{iteration}/{math.ceil(math.ceil(len(train_dataset)/args.batch)/world_size)} Loss is :{loss.item():.4f} reconstruction loss :{r_loss.item():.4f} perceptual loss :{p_loss.item():.4f}')
                if rank==0:
                    b_scans=[]
                    for index in range(min(4,args.batch)):
                        b_scans.append(recons_scans[index,0,args.save_bscans].unsqueeze(1))
                        b_scans.append(scans[index,0,args.save_bscans].unsqueeze(1))
                        # print('bscan shape ',len(b_scans),b_scans[0].shape)
                        b_scans=torch.concat(b_scans,dim=0)
                        images = [wandb.Image(bscan, caption=f"Image {i}") for i, bscan in enumerate(b_scans)]
                        wandb.log({"reconstructed bscans": images})

            epoch_loss+=loss.detach()
            recons_loss+=r_loss.detach()
            percep_loss+=p_loss.detach()

            loss.backward()

            if iteration+1%args.accum_size==0 or iteration + 1==(math.ceil(len(train_dataloader)/world_size)/args.batch):
                
                optimizer.step()
                optimizer.zero_grad()

            del scans,recons_scans
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_summary())

        epoch_loss/=math.ceil(len(train_dataset)/world_size/args.batch)
        percep_loss/=math.ceil(len(train_dataset)/world_size/args.batch)
        recons_loss/=math.ceil(len(train_dataset)/world_size/args.batch)
        
        logging.info(f"[Rank {rank}] Epoch {epoch} completed.")
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
                    val_r_loss=recons_criteron(recons_scans,scans,**recons_config)
                    
                    with torch.cuda.device(perceptual_device):
                        recons_scans = recons_scans.to(perceptual_device, non_blocking=True)
                        scans = scans.to(perceptual_device, non_blocking=True)
                        val_p_loss = perceptual_loss_fn(recons_scans.reshape(-1,1,256,256)[::args.bscan_step], scans.reshape(-1,1,256,256)[::args.bscan_step])

                    loss=val_r_loss +args.lambda_percep*val_p_loss.to(model_device)
                    
                    gathered_val_losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_val_losses, loss)
                    gathered_val_r_losses = [torch.zeros_like(val_r_loss) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_val_r_losses, val_r_loss)
                    gathered_val_p_losses = [torch.zeros_like(val_p_loss) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_val_p_losses, val_p_loss)

                    if dist.get_rank() == 0:
                        for i, (l,p,r) in enumerate(zip(gathered_val_losses,gathered_val_p_losses,gathered_val_r_losses)):
                            wandb.log({f"val loss_rank_{i}": l.item()
                                    ,f'val reconstruction loss rank {i}':r.item()
                                    ,f'val perceptual loss rank {i}':p.item()})
                    val_loss+=loss
                    val_recons_loss+=val_r_loss
                    val_percep_loss+=val_p_loss
                    
                    if args.save_recons:
                        save_recons(param_dir,epoch,recons_scans,rank,idx,args)
                       
                val_loss/=math.ceil(len(val_dataset)/world_size/args.batch)
                val_recons_loss/=math.ceil(len(val_dataset)/world_size/args.batch)
                val_percep_loss/=math.ceil(len(val_dataset)/world_size/args.batch)

                logging.info(f'Rank: {rank} VAL loss :{val_loss.item():.4f} reconstruction loss :{val_recons_loss.item():.4f} perceptual loss :{val_percep_loss.item():.4f}')
                
                if rank==0:
                    torch.save(model.state_dict(),f'{param_dir}/epoch{epoch}_val_{val_loss:.4f}_train_{epoch_loss:.4f}')

    cleanup()
    if rank==0:
        wandb.finish()

if __name__=="__main__":
    try:
    
        parser = argparse.ArgumentParser(description="train arguments")

        parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
        parser.add_argument('--batch',type=float,default=1,help='batch size')
        parser.add_argument('--epoch',type=int,default=25,help='number of epoch')
        parser.add_argument('--json',type=str,default='jsons/train_test_val_split_without_scar_with_both_res.json',help="path of json file containing path of volumes")
        parser.add_argument('--excel-path',type=str,default='d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx',help='path of excel containing labels')
        parser.add_argument('--save-dir',type=str,default='model_parameter_Resnet_pretraining')
        parser.add_argument('--accum-size',type=int,default=4,help='gradient accumulation')
        parser.add_argument('--save-freq',type=int,default=5,help='after how many epochs are the parameters saved')
        parser.add_argument('--log-dir',type=str,default='logs/Resnet_pretraining',help='the directory in which training logs are to be saved')
        parser.add_argument('--gamma',type=float,default=0.1,help='gamma for learning rate decay')
        parser.add_argument('--step-size',type=int,default=3,help='number of epochs after which learning rate is to be decayed')
        parser.add_argument('--model-path',type=str,default=None,help='path of model parameters to be loaded')
        parser.add_argument('--device',type=torch.device,default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),help='computation device')
        parser.add_argument('--log-freq',type=int,default=5,help='after how many iterations losses are logged')
        parser.add_argument('--lambda_percep',type=float,default=0.004,help="weighting factor for perceptual loss")
        parser.add_argument('--model_ch',type=list,default=[64,128,256,512],help="channels in different layers of resnet")
        parser.add_argument('--save_recons',type=bool,default=True,help="whether to save some reconstructions")
        parser.add_argument('--save_bscans',type=torch.Tensor,default=torch.tensor([36,48,60,72]),help="which reconstructed bscans to save")
        parser.add_argument('--class_dict',type=dict,default={'early':0,'inter':1,'ga':2,'wet':3,'notAMD':4})
        parser.add_argument('--percep_layer',type=str,default='relu2_2',help='which layer of vgg is to used for computation of perceptual loss')
        parser.add_argument('--bscan-step',type=int,default=2,help='step size for b-scans to be considered in perceptual loss')
        parser.add_argument('--scaling_fn',type=str,default=None,help='which scaling function to use in PixelWiseWeightedMSE')
        

        args = parser.parse_args()
        
        param_dir,log_dir=intialise_logger_nd_create_folders(args,multiprocessing=True)
        
        log_train_config(args,param_dir)
        # model=AutoEncoder(args.model_ch).to(args.device)
        # model=nn.DataParallel(model)
        logging.info(f'found {torch.cuda.device_count()} gpus!')
        # logging.info(model)
        mp.spawn(
    train,
    args=(2,args,param_dir,log_dir),
    nprocs=2,
    join=True
)

    except Exception as e:
        print(e)
        logging.error(str(e))


