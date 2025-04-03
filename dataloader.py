from torch.utils.data import Dataset
import cv2
import os 
import numpy as np
import pandas as pd
from utils.histogram import get_amdtype
import torch

def collate_fn(batch):
    lists=len(batch[0])-1
    data=[]
    for i in range(lists):
        d=torch.stack([val[i] for val in batch],dim=0)
        data.append(d)
    
    return data,torch.concat([val[-1] for val in batch],dim=0)

def undersampler(needed_samples:int,total_samples:int)->list:
    assert needed_samples>=total_samples//2,"this sampler is unfit for sampling if 2*need_samples<total_samples"
    sel_indexes=[i for i in range(0,total_samples,2)]
    nums = (2*np.random.uniform(0, total_samples//2, needed_samples-len(sel_indexes)).astype(int)+1).tolist()
    sel_indexes.extend(nums)
    assert len(sel_indexes)==needed_samples,'the sampling is not right'
    
    return sorted(sel_indexes)

    
class OCTDataset(Dataset):
    def __init__(self,image_paths,excel_path,transforms,cliplimit=1.1,**kwargs):
        super().__init__()
        self.cliplimit=cliplimit
        self.image_paths=image_paths
        self.df=pd.read_excel(excel_path,sheet_name='annotations')
            
        if 'classes' in kwargs:
            self.classes=kwargs['classes']
        else:
            self.classes={'early':0,'inter':1,'ga':2,'wet':3,'scar':4,'notAMD':5}
            
        if transforms:
            self.transforms=transforms
        else:
            self.transforms=None
            
        if 'volume_shape' in kwargs:
            self.volume_shape=kwargs['volume_shape']
        else:
            self.volume_shape=[[128,256,256],[200,256,256]]
        
        if 'undersample' in kwargs:
            self.undersample=kwargs['undersample']
        else:
            self.undersample=False

        if 'attn' in kwargs:
            self.attn=kwargs['attn']
            self.context_length=max([row[0] for row in self.volume_shape])
            
        else:
            self.attn=False

    def __getitem__(self, index):
        self.clahe=cv2.createCLAHE(clipLimit=self.cliplimit)
        scans_dir=self.image_paths[index]
        volume=[]
        sections=scans_dir.split("\\")
        pt_info=[sections[3],sections[4],sections[5]]
        amdtype=get_amdtype(pt_info,self.df)
        scan_list=sorted(os.listdir(scans_dir),key=lambda x:int(x.split("_")[-1].split(".")[0]))
        if self.undersample:
            scan_list=[scan_list[i] for i in undersampler(self.volume_shape[0],len(scan_list))]
        for bscan in scan_list:
            img=cv2.imread(scans_dir+os.sep+bscan,0)

            #histogram equalization
            img = self.clahe.apply(img)
            assert img.shape==(1024,512) or img.shape==(1024,200),f' the shape of b scan must be [1024,512] or [1024,200] but was found to be {img.shape} the scan dir is :{scans_dir}'
            if self.transforms:
                img=self.transforms(img)
            volume.append(img.squeeze(dim=0))
        volume=torch.stack(volume,dim=0)
        assert any(list(volume.shape)==vol for vol in self.volume_shape),f'the shape of scan should be {self.volume_shape} but it was found to be {volume.shape}'

        label=torch.tensor([self.classes[amdtype]])
        if self.attn:
            if volume.shape[0]<self.context_length:
                attn_mask=torch.tensor([ i>=volume.shape[0]+1 for i in range(self.context_length+1)],dtype=torch.float32)

                volume=torch.concat([volume,torch.ones(self.context_length-volume.shape[0],*self.volume_shape[0][1:])],dim=0)
                # print(torch.unique(attn_mask))
                # print(attn_mask)
            else:
                attn_mask=torch.zeros(self.context_length+1)
                # print(torch.unique(attn_mask))
                # print('context window:',self.context_length)
            assert list(attn_mask.shape)==[self.context_length+1],f'the shape of attn mask is not right:{attn_mask.shape}'
            assert any(int(torch.sum(attn_mask).item())==self.context_length-n for n in list(np.array(self.volume_shape)[:,0])),f'the attention mask is not right:{torch.sum(attn_mask)} scans dir :{scans_dir}'
            return volume.unsqueeze(dim=0),attn_mask,label

        return volume.unsqueeze(dim=0),label
            
    def __len__(self):
        return len(self.image_paths)