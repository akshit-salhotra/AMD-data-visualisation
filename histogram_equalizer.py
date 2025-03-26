import cv2
import json
import os 
import random

json_path='jsons/train_test_val_split.json'
save_dir='images/contrast_enhancement'
os.makedirs(save_dir,exist_ok=True)
with open(json_path,'r') as f:
    path_dict=json.load(f)

train_paths=path_dict['train_path']
clahe=cv2.createCLAHE(clipLimit=1.1)
num_images=len(os.listdir(save_dir))
for volume in train_paths:
    num=int(random.random()*127)
    bscan_path=volume +os.sep+os.listdir(volume)[num]

    img=cv2.imread(bscan_path,0)

    img_enhanced=clahe.apply(img)

    hstack=cv2.hconcat([img,img_enhanced])
    cv2.imshow('images',hstack)
    k=cv2.waitKey(0)
    if k==ord('q'):
        break
    elif k==ord('s'):
        cv2.imwrite(save_dir+os.sep+str(num_images)+'.jpg',hstack)
        num_images+=1
cv2.destroyAllWindows()

