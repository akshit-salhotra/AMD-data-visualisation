from dataloader import OCTDataset,collate_fn
from torch.utils.data import DataLoader
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

json_path=""
data_type=""
assert any(data_type==n for n in ['test','train']),'invalid data type'
excel_path=""
transform=transforms.Compose([[transforms.ToTensor(),
                                transforms.Resize((256,256))]])
batch=1
timeout=600

assert batch==1,'batch must be one for proper visualisation'

with open(json_path,'r') as file:
    paths=json.load(file)
    
dataset=OCTDataset(paths[f'{data_type}_path'],excel_path,transform,get_raw_scans=True)
dataloader=DataLoader(dataset,batch,shuffle=True,collate_fn=collate_fn,timeout=timeout)

for data,label in dataloader:
    volumes=data[0][0]
    raw_volumes=data[1][0]
    
    idx=torch.randint(0,volumes.shape[1],size=(4))
    
    for n,i in enumerate(range(list(idx))):
        plt.subplot(4,2,n)
        plt.imshow(raw_volumes[0,i],cmap='gray')
        plt.title('raw volume scan number',i)
        plt.axis('off')
    
    for n,i in enumerate(range(list(idx))):
        plt.subplot(4,2,4+n)
        plt.imshow(volumes[0,i],cmap='gray')
        plt.title('preprocessed volume scan number',i)
        plt.axis('off')        
        
    plt.subplots_adjust(wspace=0.5, hspace=0.3)       
    plt.show()