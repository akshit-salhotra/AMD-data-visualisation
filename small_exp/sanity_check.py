import torchvision.transforms as transforms
from dataloader import OCTDataset,collate_fn
from torch.utils.data import DataLoader
import json
import torch
from tqdm import tqdm

excel_path='excel/vol_annotations_06_03_2025.xlsx'
with open('D:\\AMD-data-visualisation\\jsons\\patient_level\\train_val_split_new_dataset.json','r') as file:
            paths= json.load(file) 
train_paths=paths['train_indices']+paths['test_indices']

transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.RandomHorizontalFlip(),
                                      transforms.Resize((256,256))])
dataset_config={'transforms':transform
                        ,'attn':False
                        ,'undersample':True
                        ,'classes':{'Early AMD':0,'Int AMD':1,'GA':2,'Wet':3,'Scar':4,"Not AMD":5}
                        ,'denoise':False
                        ,'multiThread':False
                        ,'old_excel':False}

dataset=OCTDataset(train_paths,excel_path=excel_path,**dataset_config)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False,num_workers=25,timeout=600,collate_fn=collate_fn)

if __name__=="__main__":
    count=0
    for idx,(_,_,label) in tqdm(enumerate(test_loader)):
        # print(label)
        label=label[0]
        if torch.sum(label)>1:
                if not ( label[0]==0 and label[1]==0 and label[5]==0 ):
                       print(label)
                       print(idx)
                       count+=1

    print(count)

                # print(label)