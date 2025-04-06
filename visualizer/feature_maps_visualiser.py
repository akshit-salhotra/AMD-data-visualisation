#import model
from model.resnet_3d import Resnet18_3D
#import dataloader
from dataloader import OCTDataset,collate_fn

import matplotlib.pyplot as plt
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

def hook_func(name):      
    def hook_fn(module,input,output):
        global feature_maps
        feature_maps[name]=output.detach().cpu().squeeze(dim=0)
    return hook_fn

def register_hooks(model:str,model_ob:torch.nn.Module):
    if model=='Resnet18_3D':
        layers=[f'conv{i}' for i in range(2,6)]
        
        for layer_name in layers:
            layer = getattr(model_ob, layer_name)
            layer.register_forward_hook(hook_func(layer_name))  
            
    elif model=='Seq_model_conv':
        layers=[f'seq_{i}'for i in range(4)]
        l_iter=iter(layers)
        for layer in model_ob.feature_extractor():
            if isinstance(layer,nn.Sequential):
                if hasattr(layer[0],'conv2'):
                    layer[0].conv2.register_forward_hook(hook_func(next(l_iter)))
        
    else:
        raise ValueError(f'this model is not implemented')

          
    print('registered hooks in the layers:',layers)
    
key_pressed=None
def on_key(event):
    global key_pressed
    key_pressed=event.key
    plt.close() 
    
def get_plot(data):
    ran=torch.randint(low=0,high=data.shape[0],size=(16,))
    
    fig, axs = plt.subplots(4, 4, figsize=(12, 6))
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    for n, i in enumerate(ran):
        axs[n//4, n%4].imshow(data[i], cmap='virdis')
        axs[n//4, n%4].set_title(f'feature map number  #{i.item()}')
        axs[n//4, n%4].axis('off')
    
    plt.show()
        
if __name__=="__main__":
    json_file=""
    data_type="" 
    excel_path=""
    transform=transforms.Compose([[transforms.ToTensor(),
                                transforms.Resize((256,256))]])
    num_classes=5
    batch_size=1
    timeout=600
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_features=False
    parameters_path=""
    
    assert any(data_type==n for n in ['test','train']),f"the value of data_type should be form [train,test] but it was found to be {data_type}"
    assert batch_size==1,f'batch size must be 1 for proper visualization'
    
    with open(json_file,'r') as file:
        paths=json.load(file)
    
    dataset=OCTDataset(paths[f'{data_type}_path'],excel_path,transform)
    model=Resnet18_3D(num_classes).to(device)
    model.load_state_dict(torch.load(parameters_path,device=device))
    dataloader=DataLoader(dataset,batch_size,timeout,collate_fn=collate_fn)
    feature_maps={}
    register_hooks('Seq_model_conv',model)
    
    model.eval()
    for data,label in dataloader:
        feature_maps={}
        data=[d.to(device) for d in data]
        label=label.to(device)
        
        pred=model(*data)
        
        for key,data in feature_maps.items():
            print(key)
            print(data.shape)
            
            get_plot(data)
            
            if key_pressed == 'z':
                print("Pressed 'z' — exiting loop.")
                break

            if key_pressed=='m':
                get_plot(data)
                while key_pressed=='m':
                    get_plot(data)
                    
            
            
            
        
        
        
        
        