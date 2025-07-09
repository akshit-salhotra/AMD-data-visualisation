import argparse
import matplotlib
matplotlib.use('Agg')
from utils.logger import intialise_logger_nd_create_folders
from utils.util import load_model,CrossEntropyHybrid
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset,WeightedRandomSampler
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from dataloader import OCTDataset,collate_fn
import torch.optim as optim
from model.resnet_3d import Resnet18_3D
from model.resnet_medicalnet import resnet10,resnet34,resnet50
from utils.make_plots import get_batch_stats_plot
import os
from tqdm import tqdm
import logging
import json
from sklearn.model_selection import KFold,StratifiedKFold
import torch.nn as nn
from utils.evaluation_metrics import rocPlotter
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from hooks.batch_hook import create_hook,batchnorm_stats
import wandb
import traceback


def train_step(args,iter,data,label,epoch_loss,optimizer,model,bce_criteron,ce_criteron,hybCe_criteron,dataset,num_folds):
    data=[d.to(args.device) for d in data]
    label=label.to(args.device)
    # print(image.shape)
    # with torch.no_grad():

    logits=model(*data)
        # print(logits.device,label.device)
    ce_logits=logits[:,0:2]
    ce_label=torch.argmax(label[:,0:2],dim=-1)
    ce_mask=torch.sum(label[:,0:2],dim=-1)!=0
    ce_logits=ce_logits[ce_mask]
    ce_label=ce_label[ce_mask]
    
    bce_logits=logits[:,2:5]
    bce_label=label[:,2:5]
    # print(bce_logits,ce_logits,labels)
    bce_loss=bce_criteron(bce_logits,bce_label)
    ce_loss=ce_criteron(ce_logits,ce_label)
    hyce_loss=hybCe_criteron(logits,label)

    loss=args.lambda_hy*hyce_loss+args.lambda_ce*ce_loss+args.lambda_bce*bce_loss
    # print(loss,"loss")
    # print(epoch_loss)

    epoch_loss+=loss
    if iter%5==0:
                # print(logits,label)
                wandb.log({'batch loss':loss,
                           "batch ce loss":ce_loss,
                           'batch bce loss':bce_loss,
                           'batch hybrid loss':hyce_loss
                               })
                logging.info(f'epoch:{i}/{args.epoch} iteration:{iter}/{(len(dataset)*(num_folds-1))//(num_folds*args.batch)+1} batch loss is :{loss:.4f}')
                # logging.info(f'the acc is :{torch.mean((torch.argmax(logits,dim=-1)==label).to(torch.float32)).detach().cpu()}')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return epoch_loss


def train_step_broad_only(args,iter,data,label,epoch_loss,optimizer,model,ce_criteron,dataset,num_folds):
    data=[d.to(args.device) for d in data]
    label=label.to(args.device)
    # print(image.shape)
    # with torch.no_grad():

    logits=model(*data)
        # print(logits.device,label.device)
    mapping_target=torch.tensor([0,0,1,1,1,2]).to(label.device)

    label=mapping_target[torch.argmax(label,dim=-1)]
    loss=ce_criteron(logits,label)

    # loss=args.lambda_hy*hyce_loss+args.lambda_ce*ce_loss+args.lambda_bce*bce_loss
    # print(loss,"loss")
    # print(epoch_loss)

    epoch_loss+=loss
    if iter%5==0:
                # print(logits,label)
                wandb.log({'batch loss':loss,

                               })
                logging.info(f'epoch:{i}/{args.epoch} iteration:{iter}/{(len(dataset)*(num_folds-1))//(num_folds*args.batch)+1} batch loss is :{loss:.4f}')
                # logging.info(f'the acc is :{torch.mean((torch.argmax(logits,dim=-1)==label).to(torch.float32)).detach().cpu()}')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return epoch_loss

def val_step(args,data,label,val_loss,model,bce_criteron,ce_criteron,hybCe_criteron):
    data=[d.to(args.device) for d in data]
    label=label.to(args.device)
    logits=model(*data)
    
    ce_logits=logits[:,0:2]
    ce_label=torch.argmax(label[:,0:2],dim=-1)
    bce_logits=logits[:,2:5]
    bce_label=label[:,2:5]
    bce_loss=bce_criteron(bce_logits,bce_label)
    ce_loss=ce_criteron(ce_logits,ce_label)
    hyce_loss=hybCe_criteron(logits,label).to(args.device)
   
    loss=args.lambda_hy*hyce_loss + args.lambda_ce*ce_loss+args.lambda_bce*bce_loss
    val_loss+=loss
    
    wandb.log({'batch val loss':loss,
               'batch val hy loss':hyce_loss,
               'batch val ce loss':ce_loss,
               'batch val bce loss':bce_loss})
    # preds=torch.argmax(logits,dim=-1)
    # assert label.shape==preds.shape, 'the number of labels and images do not match'
    return val_loss,logits,label

def val_step_broad_only(args,data,label,val_loss,model,ce_criteron):
    data=[d.to(args.device) for d in data]
    label=label.to(args.device)
    logits=model(*data)

    mapping_target=torch.tensor([0,0,1,1,1,2]).to(label.device)

    label=mapping_target[torch.argmax(label,dim=-1)]   

    ce_loss=ce_criteron(logits,label)
   
    # loss=args.lambda_hy*hyce_loss + args.lambda_ce*ce_loss+args.lambda_bce*bce_loss
    val_loss+=ce_loss
    
    wandb.log({'batch val loss':ce_loss,
  })
    # preds=torch.argmax(logits,dim=-1)
    # assert label.shape==preds.shape, 'the number of labels and images do not match'
    return val_loss,logits,label






if __name__=="__main__":
    try:
    
        parser = argparse.ArgumentParser(description="train arguments")

        parser.add_argument("--lr", type=float, default=0.000025, help="learning rate")
        parser.add_argument('--batch',type=float,default=12,help='batch size')
        parser.add_argument('--epoch',type=int,default=10,help='number of epoch')
        parser.add_argument('--json',type=str,default='D:\\AMD-data-visualisation\\jsons\\patient_level\\train_val_split_new_dataset.json',help="path of json file containing path of volumes")
        parser.add_argument('--excel-path',type=str,default='excel/vol_annotations_06_03_2025.xlsx',help='path of excel containing labels')
        parser.add_argument('--save-dir',type=str,default='model_parameter_Resnet_medicalnet')
        parser.add_argument("--eval_freq",type=int,default=10,help="eval freq")
        parser.add_argument('--save-freq',type=int,default=5,help='after how many epochs are the parameters saved')
        parser.add_argument('--log-dir',type=str,default='logs/Resnet_medicalnet',help='the directory in which training logs are to be saved')
        parser.add_argument('--gamma',type=float,default=0.1,help='gamma for learning rate decay')
        parser.add_argument('--step-size',type=int,default=10,help='number of epochs after which learning rate is to be decayed')
        parser.add_argument('--model-path',type=str,default="pretrained\\resnet_34_23dataset.pth",help='path of model parameters to be loaded')
        parser.add_argument('--device',type=torch.device,default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),help='computation device')
        parser.add_argument('--weight_ce',type=torch.tensor,default=torch.tensor([0.2,0.2,1]),help='weights for weighted cross entropy')
        parser.add_argument('--weight_bce',type=torch.tensor,default=torch.tensor([3.96,4.95,0.59]),help='weights for weighted cross entropy')
        parser.add_argument('--weight_broad',type=torch.tensor,default=torch.tensor([0.5,0.4,0.4,0.33,1.25,1]))
        parser.add_argument('--sampling_prob',type=dict,default={0:0.25,1:0.05,2:0.5})
        parser.add_argument('--num_samples_per_epoch',type=int,default=1000)
        parser.add_argument('--class_dict',type=dict,default={'Early AMD':0,'Int AMD':1,'GA':2,'Wet':3,'Scar':4,"Not AMD":5})
        parser.add_argument('--num_classes',type=int,default=3,help="number of classes of the classifier")
        parser.add_argument('--lambda_bce',type=float,default=1.0,help='weighting factor for the bce loss')
        parser.add_argument('--lambda_ce',type=float,default=1.0)
        parser.add_argument("--lambda_hy",type=float,default=4)
        parser.add_argument('--targets_path',type=str,default="targets.npy")
        
        # parser.add_argument('--model_ch',type=list,default=[16,32,64,128],help="channels in different layers of resnet")

        args = parser.parse_args()
        
        run=wandb.init(project='classifier AMD'
        ,config=vars(args))
        # wandb.init(id="rosy-dew-31", resume="must")
        param_dir=intialise_logger_nd_create_folders(args)
        
        # device_ids=[1,2,3]
        register_hooks=False
        # model=Resnet18_3D(num_classes=args.num_classes,ch=args.model_ch).to(args.device)
        model=resnet34(num_classes=args.num_classes,shortcut_type='A').to(args.device)
        arch="resnet34_medicalnet"
        multiLabel=True

        print("the architecure is :",arch)
        print('multi label is :',multiLabel)

        run.config.update({'arch':arch
                   ,'multiLabel':multiLabel})
        
        wandb.watch(model,log='all',log_freq=5)
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Convert all BatchNorm layers
        # model=Seq_Model(num_classes=args.num_classes,device=args.device).to(args.device)
        model=nn.DataParallel(model)
        logging.info(f"the wandb run name is : {wandb.run.name}")
        logging.info(f'found {torch.cuda.device_count()} gpus!')
        logging.info(model)
        # criteron=CrossEntropyLoss(weight=args.weight_matrix.to(args.device))##make sure to add weight factor
        # bce_criteron=BCEWithLogitsLoss(pos_weight=args.weight_bce.to(args.device))
        # ce_criteron=CrossEntropyLoss(weight=args.weight_ce.to(args.device))
        # hybCe_criteron=CrossEntropyHybrid(weights=args.weight_broad.to(args.device)).to(args.device)
        ce_criteron=CrossEntropyLoss(weight=args.weight_ce.to(args.device))
        #early,inter,ga,wet,scar,notamd
        optimizer=optim.Adam(model.parameters(),args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        
        if args.model_path:
            load_model(args,model)
            
        with open(args.json,'r') as file:
            paths= json.load(file) 
        train_paths=paths['train_indices']

        transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.RandomHorizontalFlip(),
                                      transforms.Resize((256,256))])
        dataset_config={'transforms':transform
                        ,'attn':False
                        ,'undersample':True
                        ,'classes':args.class_dict
                        ,'denoise':False
                        ,'multiThread':False
                        ,'old_excel':False}
        
        run.config.update(dataset_config)
        dataset=OCTDataset(train_paths,args.excel_path,**dataset_config)
        # dataset=Subset(dataset,range(40))
        num_folds=5
        # kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        kf=StratifiedKFold(num_folds,shuffle=True,random_state=42)

        #registering hooks
        
        if register_hooks:
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    hooks.append(module.register_forward_hook(create_hook(name)))
        # print(dataset[0])
        # get_target_partial = partial(get_target, dataset=dataset)
        
        # with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust worker count as needed
                    # targets_list = list(tqdm(executor.map(get_target_partial, range(len(dataset))), total=len(dataset)))
        if os.path.exists(args.targets_path):
             targets=torch.from_numpy(np.load(args.targets_path)).to(args.device)
             print("targets loaded from",args.targets_path)
        else:
            targets_list=[dataset[i][1] for i in tqdm(range(len(dataset)))]
            targets=torch.stack(targets_list,dim=0).to(args.device)
            
            mapping_target=torch.tensor([0,0,1,1,1,2]).to(args.device)

            targets=mapping_target[torch.argmax(targets,dim=-1)]

            np.save(args.targets_path,targets.detach().cpu().numpy())

        for fold, (train_idx, val_idx) in enumerate(kf.split(X=targets.cpu().detach(),y=targets.cpu().detach())):
        # for fold in range(1):

            logging.info(f'fold number:{str(fold)}')
            print(f'fold number :',fold)
            
            # for _,_,t in range(len(Subset(dataset,train_idx))):
            # print(dataset[0])
            # print(targets)
            # print(args.sampling_prob)
            # print(train_idx)
            weights=[args.sampling_prob.get(targets[i].item()) for i in train_idx]
            # print(weights)
            train_sampler=WeightedRandomSampler(weights,args.num_samples_per_epoch,replacement=False)
            train_loader = DataLoader(Subset(dataset,train_idx), batch_size=args.batch, sampler=train_sampler,num_workers=15,persistent_workers=True,timeout=600,collate_fn=collate_fn)
            test_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch, shuffle=False,num_workers=15,persistent_workers=True,timeout=600,collate_fn=collate_fn)
            if args.model_path and re.search(r'fold(\d+)',args.model_path):
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
                    # model.eval()
                    preds=[]
                    labels=[]
                    for iter,(data,_,label) in tqdm(enumerate(train_loader)):
                        # print(image.shape)
                        # print(label)
                        epoch_loss=train_step_broad_only(args,iter,data,label,epoch_loss,optimizer,model,ce_criteron,dataset,num_folds)
                        # preds.extend(pred.detach().cpu().tolist())
                        # labels.extend(label.detach().cpu().tolist())
                    # print(epoch_loss)
                    # print(((len(dataset)*(num_folds-1))//(num_folds*args.batch)+1))
                    epoch_loss/=((len(dataset)*(num_folds-1))//(num_folds*args.batch)+1)
                    # cm=confusion_matrix(labels,preds)
                    # classes=["early","inter","ga","wet","notamd"]
                   
                    # plt.figure(figsize=(6, 4))
                    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=classes,yticklabels=classes)
                    # plt.title(f'Confusion Matrix,acc:{np.mean(np.array(labels)==np.array(preds)):.4f}')
                    # plt.xlabel('Predicted')
                    # plt.ylabel('Actual')

                    # # Save as image
                    # plt.savefig(f"{param_dir+os.sep}confusion_matrix_fold_{fold}_epoch_{i}", dpi=300)
                    # plt.close()
                    logging.info(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f}')
                    pbar.set_postfix({'average epoch loss':f'{epoch_loss:.4f}'})
                    scheduler.step()
                    
                    if i%args.eval_freq==0 or args.epoch==i+1:
                        val_loss=0
                        correct_pred=0
                        model.eval()
                        val_labels=[]
                        val_preds=[]
                        with torch.no_grad():
                            for (data,_,label) in test_loader:
                                # print(data,label)
                                val_loss,val_logit,val_label=val_step_broad_only(args,data,label,val_loss,model,ce_criteron)
                                # val_preds=nn.functional.sigmoid(val)
                                # print(preds.device,label.device)
                                # correct_pred+=(label==preds).sum().item()
                                val_labels.extend(val_label.detach().cpu().tolist())
                                val_preds.extend(val_logit.detach().cpu().tolist())

                            # logging.info(f'val accuarcy is : {correct_pred/(len(dataset)/num_folds):.4f}')
                            val_loss/=(len(dataset)//(num_folds*args.batch)+1)
                            logging.info(f'VAL loss :{val_loss:.4f}')
                            val_labels=np.array(val_labels)
                            val_preds=np.array(val_preds)

                            broad_preds=np.argmax(val_preds,axis=-1)

                            # broad_preds=np.argmax(np.concatenate([np.max(val_preds[:,0:2],axis=-1,keepdims=True),np.max(val_preds[:,2:5],axis=-1,keepdims=True),np.expand_dims(val_preds[:,5],axis=-1)],axis=-1),axis=-1)
                            # broad_labels=np.concatenate([np.max(val_labels[:,0:2],axis=-1,keepdims=True),np.max(val_labels[:,2:5],axis=-1,keepdims=True),np.expand_dims(val_labels[:,5],axis=-1)],axis=-1)
                            
                            # print(broad_labels)
                            # broad_labels=np.argmax(broad_labels,axis=-1)
                            broad_labels=val_labels

                            # late_preds=nn.functional.sigmoid(torch.from_numpy(val_preds[:,2:5])).cpu().detach().numpy()
                            # late_labels=val_labels[:,2:5]

                            # rocPlotter(broad_preds,)
                            cm=confusion_matrix(broad_labels,broad_preds)
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['early','late','not'],yticklabels=['early','late','not'])
                            plt.title(f'Confusion Matrix,acc:{np.mean(broad_preds==broad_labels):.4f}')
                            plt.xlabel('Predicted')
                            plt.ylabel('Actual')
                            wandb.log({'val loss':val_loss
                                       })
                            # rocPlotter()
                            plt.savefig(f'{param_dir+os.sep}confusion_matrix_val_fold_{fold}_epoch_{i}.png')
                            # rocPlotter(late_preds,late_labels,3,f'{param_dir+os.sep}roc_val_fold_{fold}_epoch_{i}.png',None,classNames=['GA','Wet','Scar'] )
                            # plt.close()
                            plt.clf()
                            torch.save(model.state_dict(),f'{param_dir}/fold{fold}_epoch{i}_val_{val_loss:.4f}_train_{epoch_loss:.4f}')
                            # batch_save="images//plots//batch_statistics"
                            # os.makedirs(batch_save,exist_ok=True)
                            # get_batch_stats_plot(batchnorm_stats,f'{batch_save+os.sep}fold_{fold}_epoch_{i}')
                    elif i% args.save_freq==0:
                            torch.save(model.state_dict(),f'{param_dir}/fold{fold}_epoch{i}_train_{epoch_loss:.4f}')

                         
                
        print('training completed !!')
        logging.info('training complete!!!')
        wandb.finish()
    
    except Exception as e:
        print(e)
        logging.error(str(e))
        traceback.print_exc()
    