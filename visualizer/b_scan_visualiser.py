from dataloader import OCTDataset,collate_fn
from torch.utils.data import DataLoader
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

json_path="jsons/train_test_val_split_without_scar.json"
data_type="train"
assert any(data_type==n for n in ['test','train']),'invalid data type'
excel_path="d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256,256))])
batch=1
# timeout=600

assert batch==1,'batch must be one for proper visualisation'

with open(json_path,'r') as file:
    paths=json.load(file)
    
dataset=OCTDataset(paths[f'{data_type}_path'],excel_path,transform,get_raw_scans=True)
dataloader=DataLoader(dataset,batch,shuffle=True,collate_fn=collate_fn)

key_pressed=None
def on_key(event):
    global key_pressed
    key_pressed=event.key
    plt.close()

def get_plots(raw_volumes,volumes):

    idx=torch.randint(0,int(volumes.shape[1]),(4,))
    
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    for n, i in enumerate(idx):
        axs[0, n].imshow(raw_volumes[0, i], cmap='gray')
        axs[0, n].set_title(f'Raw scan #{i.item()}')
        axs[0, n].axis('off')

    for n, i in enumerate(idx):
        axs[1, n].imshow(volumes[0, i], cmap='gray')
        axs[1, n].set_title(f'Preprocessed #{i.item()}')
        axs[1, n].axis('off')

    # plt.tight_layout()
    plt.show()

for data,label in dataloader:
    volumes=data[0][0]
    raw_volumes=data[1][0]
    print(volumes.shape)
    print(label,dataset.classes)
    get_plots(raw_volumes,volumes)
    if key_pressed == 'z':
        print("Pressed 'z' — exiting loop.")
        break

    if key_pressed=='m':
        get_plots(raw_volumes,volumes)
        while key_pressed=='m':
            get_plots(raw_volumes,volumes)
