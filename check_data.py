import torch
# import training_classifier
# import evaluating_classifier
from model.resnet_3d import Resnet18_3D
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from dataloader import OCTDataset,collate_fn
import torchvision.transforms as transforms
import json
from torch.utils.data import DataLoader
json_path="jsons\\train_test_val_split_without_scar.json"
with open(json_path,'r') as file:
        paths=json.load(file)
transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.RandomHorizontalFlip(),
                                      transforms.Resize((256,256))])

dataset=OCTDataset(paths['train_path'],"d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx",transform,attn=False,undersample=False,classes={'early':0,'inter':1,'ga':2,'wet':3,'notAMD':4})

model=Resnet18_3D(5)
model.load_state_dict(torch.load("model_parameter_Resnet\\13\\fold0_epoch24_val_0.2727_train_0.2727"))
model=nn.DataParallel(model)
# print()
preds=[]
labels=[]
train_loader=DataLoader(dataset,32,shuffle=False,num_workers=8,timeout=600,collate_fn=collate_fn)
for data,label in train_loader:
    data=[d.to(torch.device('cuda')) for d in data]
    label=label.to('cuda')
    logits=model(*data)
    pred=torch.argmax(logits,dim=-1)
    preds.extend(pred.detach().cpu().tolist())
    labels.extend(label.detach().cpu().tolist())

cm=confusion_matrix(labels,preds)
classes=["early","inter","ga","wet","notamd"]
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=classes,yticklabels=classes)
plt.title(f'Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.savefig('confusion_metrics.png')


