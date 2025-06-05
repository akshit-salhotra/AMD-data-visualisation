from utils.util import SliceLevelPerceptualLoss
import torch
from dataloader import OCTDataset,collate_fn
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

vgg=SliceLevelPerceptualLoss()


# model_path="model_parameter_Resnet_medicalnet\\7\\fold4_epoch14_val_0.3108_train_0.3457"
transform=transforms.Compose([transforms.ToTensor(),
                        #   transforms.RandomHorizontalFlip(),
                            transforms.Resize((256,256))])
device='cuda' if torch.cuda.is_available() else 'cpu'
json_path="jsons\\train_test_val_split_without_scar_with_both_res.json"
excel_path=r"d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
num_slices=4

with open(json_path,'r') as f:
    paths=json.load(f)

dataset=OCTDataset(paths['json_path'],excel_path,transform)
dataloader=DataLoader(dataset,1,True,num_workers=4,timeout=600,collate_fn=collate_fn,undersample=True)

for data,_,_ in dataloader:
    data=data[0].to(device)
    data=data.view(1,-1,256,256)
    feature_maps=vgg.get_feature_map(data)
    
    sample = random.sample([i for i in range(data.shape[1])], num_slices)
    
    for i,idx in enumerate(sample):
        plt.subplot(num_slices,2,2*(i+1))
        plt.imshow(feature_maps.squeeze()[idx],cmap='viridis')
        plt.subplot(num_slices,2,2*(i+1)+1)
        plt.imshow(data[0,idx],cmap='gray')
    plt.show()
    