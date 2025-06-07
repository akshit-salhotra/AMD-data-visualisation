import torch
import torch.nn as nn
from model.resnet_medicalnet import resnet10
from dataloader import OCTDataset,collate_fn
import matplotlib.pyplot as plt
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os 
conv_activations=None
gradients=None
batch=1

def hook_fn(module,input,output):
    global conv_activations
    conv_activations=output
    print('update activations')
    

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]
    print('updated gradients')
    
assert batch==1,"batch must be one"
model_path="model_parameter_Resnet_medicalnet\\7\\fold4_epoch14_val_0.3108_train_0.3457"
save_folder="images\\grad_cam"
os.makedirs(save_folder,exist_ok=True)
num_classes=5
num_workers=4
timeout=600
num_bscans_visualised=2
assert num_bscans_visualised%2==0,"num of b scans to be visualised must be a multiple of 4"
data_indices=[1,10,15,21,17,31,47,89,115,32,5,85,74,66,55,44,51]
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
excel_path="d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
json_path="jsons/train_test_val_split_without_scar_with_both_res.json"
data='test'
assert data=='test' or 'train',"data can either be test or train"
transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])

with open(json_path,'r') as f:
    paths=json.load(f)

dataset=OCTDataset(paths[f'{data}_path'],excel_path,undersample=True,transforms=transform)
# dataloader=DataLoader(dataset,batch,shuffle=False,num_workers=8,timeout=timeout,collate_fn=collate_fn)

model=resnet10(num_classes=num_classes).to(device)
new_params={}
params=torch.load(model_path)
for key in params.keys():
     new_params[key[7:]]=params.get(key)

model.load_state_dict(new_params)
print('model weights loaded successfully. path:',model_path)

for module in reversed(list(model.modules())):
    if isinstance(module, nn.ReLU):
        module.register_forward_hook(hook_fn)
        module.register_backward_hook(backward_hook)
        print('hook registered successfully....')
        break
        
model.eval()
# with torch.no_grad():       
for i in data_indices:
            vol,label=dataset[i]
            vol=vol.to(device).unsqueeze(dim=0)
            label=label.to(device).squeeze(dim=0)
            
            logits=model(vol)
            pred=torch.argmax(logits,dim=-1)
            # print(pred.requires_grad)
            logits[0][pred].backward()
            conv_activations=conv_activations.squeeze()
            gradients=gradients.squeeze()
            # print("shapes")
            # print(conv_activations.shape,gradients.shape)
            weights = gradients.mean(dim=(1, 2,3))
            # print(weights.shape,conv_activations.shape)  
            # print(torch.unique(weights))              
            cam = torch.sum(weights.unsqueeze(1).unsqueeze(2).unsqueeze(3) * conv_activations, dim=0) 
            print('unqiue values are :')
            # print(torch.unique(cam),cam.shape)
            cam = F.relu(cam)
            print(torch.unique(cam))                            
            cam = cam - cam.min()
            if cam.max()==0:
                 print("max value is zero")
                 continue
            cam = cam / cam.max()
            # print(cam.shape)
            cam=F.interpolate(cam.unsqueeze(0).unsqueeze(0),scale_factor=(8.0,8.0,8.0),mode='trilinear').squeeze(dim=(0,1))
            
            assert cam.shape==(128,256,256),f'the shape of heat map is not right ,{cam.shape}'
            
            mean_importance=torch.mean(cam,dim=(1,2))
            # print(mean_importance)
            _, top_indices = torch.topk(mean_importance, k=num_bscans_visualised)
            
            print('predicted class:',pred)
            print('actual class',label)
            print('max values of b scans is :')
            for count,i in enumerate(top_indices):
                plt.subplot(num_bscans_visualised//2,4,2*count+1)
                print(cam[i].max())
                bscan_mask=cam[i]-cam.min()
                bscan_mask=bscan_mask/bscan_mask.max()
                plt.imshow(bscan_mask.detach().cpu().numpy(),cmap='viridis')
                # plt.colorbar()
                plt.subplot(num_bscans_visualised//2,4,2*count+2)
                plt.imshow(vol[0][0][i].detach().cpu().numpy(),cmap='gray')
            plt.suptitle(f'predicted class :{pred} actual class :{label}')
            plt.savefig(save_folder+os.sep+f"json_{os.path.basename(json_path)}_{data}_set_index_{i}.png")
                
                
            
                