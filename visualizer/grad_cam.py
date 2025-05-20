import torch
import torch.nn as nn
from model.resnet_medicalnet import resnet10
from dataloader import OCTDataset,collate_fn
import matplotlib.pyplot as plt
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
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
model_path=None
num_classes=5
num_workers=8
timeout=600
num_bscans_visualised=8
assert num_bscans_visualised%4==0,"num of b scans to be visualised must be a multiple of 4"
data_indices=[]
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
excel_path="d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
json_path="jsons/train_test_val_split_without_scar_with_both_res.json"
transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])

with open(json_path,'r') as f:
    paths=json.load(f)

dataset=OCTDataset(paths['test_paths'],excel_path,transforms=transform)
# dataloader=DataLoader(dataset,batch,shuffle=False,num_workers=8,timeout=timeout,collate_fn=collate_fn)

model=resnet10(num_classes=num_classes).to(device)

model.load_state_dict(torch.load(model_path))
print('model weights loaded successfully. path:',model_path)

for module in reversed(list(model.modules())):
    if isinstance(module, nn.ReLU):
        module.register_forward_hook(hook_fn)
        module.register_backward_hook(backward_hook)
        print('hook registered successfully....')
        
        
        
for i in data_indices:
        vol,label=dataset[i]
        vol=vol.to(device).unsqueeze(dim=0)
        label=label.to(device).squeeze(dim=0)
        
        logits=model(vol)
        pred=torch.argmax(logits,dim=-1)
        pred.backward()
        conv_activations=conv_activations.squeeze()
        gradients=gradients.squeeze()
        weights = gradients.mean(dim=(1, 2,3))                
        cam = torch.sum(weights * conv_activations, dim=0) 
        cam = F.relu(cam)                               
        cam = cam - cam.min()
        cam = cam / cam.max() 
        cam=F.interpolate(cam.unsqueeze(dim=0),scale_factor=(8.0,8.0,8.0),mode='bilinear').squeeze()
        
        assert cam.shape==(128,256,256),f'the shape of heat map is not right ,{cam.shape}'
        
        mean_importance=torch.mean(cam,dim=(1,2))
        _, top_indices = torch.topk(mean_importance, k=num_bscans_visualised)
        
        print('predicted class:',pred)
        print('actual class',label)
        print('max values of b scans is :')
        for i in top_indices:
            plt.subplot(num_bscans_visualised//2,4,i+1)
            print(bscan_mask.max())
            bscan_mask=cam[i]-cam.min()
            bscan_mask=bscan_mask/bscan_mask.max()
            plt.imshow(bscan_mask,cmap='viridis')
            plt.colorbar()
            plt.subplot(num_bscans_visualised//2,4,i+2)
            plt.imshow(vol[i],cmap='greys')
            
            
        
            