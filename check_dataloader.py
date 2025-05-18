from dataloader import OCTDataset,collate_fn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import json

json_path='jsons/train_test_val_split.json'
excel_path='d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx'
with open(json_path,'r') as f:
    paths=json.load(f)
transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.RandomHorizontalFlip(),
                                      transforms.Resize((256,256))])
dataset=OCTDataset(paths['train_path'],excel_path,transform)
dataloader=DataLoader(dataset,batch_size=6,shuffle=True,collate_fn=collate_fn)

for a,label in dataloader:
    print(label)