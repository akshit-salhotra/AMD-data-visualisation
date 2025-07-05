import torch
import logging
import torch.nn as nn
import torchvision.models as models
import wandb
import torch.nn.functional as f
import random

def load_model(args,model):
    params=torch.load(args.model_path)['state_dict']

    # for key in params.keys():
    #      if key.startswith('module.'):
    #           params[f'module.{key}']=params.pop(key)
        
    missing,unexpected=model.load_state_dict(params,strict=False)
    logging.info(f'the missing keys are : \n{missing}')
    logging.info(f'the unexpected keys are :\n{unexpected}')
    print('loaded model parameters from ',args.model_path)
    logging.info(f'loaded model parameters from {args.model_path}')
    
    
class SliceLevelPerceptualLoss(nn.Module):
    def __init__(self, layer='relu3_3'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        self.blocks = nn.Sequential()

        layer_map = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23
        }

        self.blocks = vgg[:layer_map[layer] + 1].eval()
        for param in self.blocks.parameters():
            param.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def forward(self, x, y):
        '''
        x and y be bscan level entities
        '''
        assert x.device==y.device, f' the device of x and y is not same'
        # Normalize inputs to match VGG expectations
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)

        self.blocks.to(x.device)
        # Compute perceptual features
        fx = self.blocks(x)
        fy = self.blocks(y)

        return nn.functional.mse_loss(fx, fy)
    
    def get_feature_map(self,x):

        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        self.blocks.to(x.device)
        fx = self.blocks(x)
        
        print(fx.shape)
        
        return fx

class PixelWiseWeightedMSE(nn.Module):
    def __init__(self,scaling_function='Hyb_Sigmoid'):
        super().__init__()
        self.scaleFuncDict={'Hyb_Sigmoid':self.Hybrid_Sigmoid
                            ,'adaptiveHybridSigmoid':self.adaptiveHybridSigmoid}
        self.scaleFunc=scaling_function
    
    def forward(self,pred,target,**kwargs):

        assert pred.shape==target.shape , f' the shape of prediction and target must be same'

        weights=self.scaleFuncDict[self.scaleFunc](target,**kwargs)
        loss=torch.sum(weights*(pred-target)**2)/(torch.numel(pred))

        return loss
    
    def Hybrid_Sigmoid(self,target,k=1.5,theta=0.5):
        return 1 / (1 + torch.exp(-k * (target - theta)))
    
    def adaptiveHybridSigmoid(self,target,iteration,rank,delta,intial_k=0,theta=0.4,log=True):
        k=intial_k+delta*iteration

        if rank==0:

            wandb.log({"k (activation function)":k})
        return 1/(1+torch.exp(-k*(target-theta)))

class CrossEntropyHybrid(nn.Module):

    def __init__(self,weights:torch.Tensor):
        super().__init__()
        assert weights.shape[0]==6 
        self.weights=weights

        self.register_buffer('dummy',torch.zeros(1))

    def forward(self,preds:torch.Tensor,targets:torch.Tensor):
        '''
        preds:(batch,n_classes) Logits of all the classes
        targets:(batch,n_classes) target of all the classes

        '''

        assert self.weights.shape[0]==preds.shape[1],"the shape of preds is not right , it must include all the logits not just the ones involved in softmax"

        preds_early,early_idx=torch.max(preds[:,0:2],dim=-1,keepdims=True)
        preds_late,late_idx=torch.max(preds[:,2:5],dim=-1,keepdims=True)

        early_idx=early_idx.squeeze(-1)
        late_idx=late_idx.squeeze(-1)

        # weights=torch.tensor([self.weights[i].item() for i in targets]).to(self.dummy.device)#(batch)

        weights=torch.tensor([self.weights[torch.argmax(t)].item() if torch.sum(t)==1 else self.weights[2+late_idx[i]]  for i,t in enumerate(targets)]).to(self.dummy.device)#(batch)

        forced_preds=[]
        forced_target=[]
        forced_weights=[]

        for p,t,e,l,pe,pl in zip(preds,targets,early_idx,late_idx,preds_early,preds_late):
            # if torch.sum(t)>1:
            ls=[[pe,p[i],p[-1]] for i in range(2,5) if (t[i]==1 and i!=l)]

            if len(ls):
                forced_preds.extend(ls)
                forced_target.extend([t]*len(ls))
                forced_weights.extend([self.weights[i] for i in range(2,5) if (t[i]==1 and i!=l)])

            if t[e]==0:
                forced_preds.append([p[1 if e==0 else 0].item(),pl.item(),p[-1].item()])
                forced_target.append(t)
                forced_weights.append(self.weights[1 if e==0 else 0])


        preds=torch.concat([preds_early,preds_late,preds[:,5].unsqueeze(dim=-1)],dim=-1)

        # print(weights.shape)

        # weights=torch.concat([torch.tensor([[self.weights[i]]for i in idx_early]),torch.tensor([[self.weights[i+2]]for i in idx_late]),torch.ones(preds.shape[0],1)*weights[5]],dim=-1)

        # log_preds=f.log_softmax(preds,dim=-1) #(batch,3)
        # print(log_preds.shape)
        probs=f.softmax(preds,dim=-1)
        # print(probs.shape,'probs')

        '''
        invert the prob of wrong scans &then append the forces ones and then we should be good to go
        '''

        batch_size = targets.shape[0]
        rows = torch.arange(batch_size).to(self.dummy.device)

        mask = torch.zeros_like(probs, dtype=torch.bool).to(self.dummy.device)

        cond_early = targets[rows, early_idx] != 1
        cond_late = targets[rows, late_idx] != 1

        # print(mask.device,rows.device,early_idx.device,cond_early.device)
        mask[rows[cond_early], early_idx[cond_early]] = True
        mask[rows[cond_late], late_idx[cond_late]] = True

        probs = probs.clone()
        probs[mask] = 1 - probs[mask]
        log_preds = torch.log(probs)

        forced_target=torch.stack(forced_target,dim=0)
        forced_preds=torch.log_softmax(torch.tensor(forced_preds,device=self.dummy.device),dim=-1)


        combined_targets=torch.concat([targets,forced_target],dim=0)
        combined_preds=torch.concat([log_preds,forced_preds],dim=0)
        weights=torch.concat([weights,torch.tensor(forced_weights,device=self.dummy.device)],dim=0)

        mapping_target=torch.tensor([0,0,1,1,1,2]).to(self.dummy.device)

        combined_targets=mapping_target[torch.argmax(combined_targets,dim=-1)]
        # print(targets,weights)
        unweighted_loss=f.nll_loss(combined_preds,combined_targets,reduction='none')#(batch,1)


        assert unweighted_loss.shape[0]==combined_targets.shape[0] ,f'{unweighted_loss.shape}'
        assert unweighted_loss.shape[0]==weights.shape[0],f'weights :{weights.shape} unweighted loss :{unweighted_loss.shape}'

        loss=torch.mean(unweighted_loss*weights)

        return loss


def inferMultiLabelHybrid(logits:torch.Tensor,thres:torch.Tensor):
    '''
    logits:(n,n_classes):logits of all classes
    thres:(3,) , the list of thresholds for multi-label classification of late amd

    returns
    broad_preds:(n,):preds for broad level classification
    early_preds:(n,):preds for early AMD
    late_preds:(n,3):preds for latter stage AMD

    returns -1(early_preds) for a particular scan , if the scan does not belong to early_preds
    '''

    preds_early,_=torch.max(logits[:,0:2],dim=-1,keepdims=True)
    preds_late,_=torch.max(logits[:,2:5],dim=-1,keepdims=True)

    broad_preds=torch.argmax(torch.concat([preds_early,preds_late,logits[:,5].unsqueeze(dim=-1)],dim=-1),dim=-1)

    early_preds=torch.ones_like(broad_preds)*-1
    early_mask=broad_preds==0
    late_mask=broad_preds==1
    early_preds[early_mask]=torch.argmax(logits[:,:2],dim=-1)

    late_preds=((logits[:,2:5]>thres) & late_mask.unsqueeze(dim=-1)).int()

    #if no logit is above thres , even after the scan being classified late -> assign the scan to class with max logit
    none_mask=late_mask & (torch.sum(late_preds,dim=-1)==0)
    late_preds[none_mask]=f.one_hot(torch.argmax(logits[:,2:5][none_mask],dim=-1),3)

    return broad_preds,early_preds,late_preds




if __name__ == "__main__":
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ce=CrossEntropyHybrid(torch.tensor([0.3,0.5,0.2,0.56,0.1,0.4],device='cuda')).to(device)

    preds=torch.randint(-100,100,(12,6)).float().to(device)

    target=torch.randint(0,2,(12,6)).to(device)

    print(ce(preds,target))




        