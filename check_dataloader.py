from dataloader import OCTDataset,collate_fn,B_ScanDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import json

json_path='jsons/train_test_val_split.json'
excel_path='excel/vol_annotations_06_03_2025.xlsx'
with open(json_path,'r') as f:
    paths=json.load(f)
transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.RandomHorizontalFlip(),
                                      transforms.Resize((256,256))])
dataset=B_ScanDataset(transform,excel_path,False)
collate_fn=None
# dataset=OCTDataset(paths['train_path'],excel_path,transform)
dataloader=DataLoader(dataset,batch_size=6,shuffle=True,collate_fn=collate_fn)

for a in tqdm(dataloader):
    # print(label)
    pass