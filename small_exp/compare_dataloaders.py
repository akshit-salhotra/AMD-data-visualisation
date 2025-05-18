import torch
from dataloader import OCTDataset,collate_fn
from small_exp.dataloader import OCTDataset1 as OLD
import torchvision.transforms as transforms
import json
from torch.utils.data import DataLoader


if __name__=="__main__":
    # model_path="model_parameter_Resnet3D\\15\\fold4_epoch30_val_0.1482_train_0.0672"
    device='cuda' if torch.cuda.is_available() else 'cpu'
    json_path="jsons\\train_test_val_split.json"
    excel_path=r"d:\\cleaning_GUI_annotated_data\\tab_data_annotated_pats.xlsx"
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])
    batch_size=1
    num_workers=8
    timeout=600
    # results_dir="results"
    with open(json_path,'r') as file:
            paths=json.load(file)
    # os.makedirs(results_dir,exist_ok=True)
    # save_file= model_path.split(os.sep)[-2]+"_"+model_path.split(os.sep)[-1]+'confusion_matrix_test_set.png'
    old_d=OLD(paths['test_path'],excel_path,transform)
    new_d=OCTDataset(paths['test_path'],excel_path,transform)

    new_d=DataLoader(new_d,batch_size=12,shuffle=False,num_workers=num_workers,timeout=timeout,collate_fn=collate_fn)
    old_d=DataLoader(old_d,batch_size=12,shuffle=False,num_workers=num_workers,timeout=timeout)

    for ((data_n,l),(data_o,l2)) in zip(new_d,old_d):
            # print(data_n.shape,l.shape,data_o.shape,l2.shape)
            # print(l2)
        #     print('hi')
            print(torch.eq(data_n[0],data_o).all().item(), "assertion failed")
            assert torch.eq(l,l2).all().item(),'labels not right'

    print('completed')