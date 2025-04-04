import cupy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import numpy as np

print('imports complete ')
bscan_dir=r"d:\cleaning_GUI_annotated_Data\Cirrus_OCT_Imaging_Data\000003162\L\20080818\124239\OPT\Carl_Zeiss_Meditec\200X1024X200\Original\B-Scans"
images=sorted(os.listdir(bscan_dir),key=lambda x:int(x.split("_")[-1].split('.')[0].strip()))
# print(images)
volume=[]

for i in range(0,len(images)):
    image_path=os.path.join(bscan_dir+os.sep+images[i])
    image=cv2.imread(image_path,0)
    cv2.imshow('bscan',image)
    cv2.waitKey(1)
    image=cv2.resize(image,(image.shape[0]//2,image.shape[1]//2))
    # print(image.shape)
    if i%50==0:
        volume.append(image.tolist())
volume=cp.array(volume)/255
cv2.destroyAllWindows()
# Visualize a cross-section from different angles
fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(131, projection='3d')
ax1.voxels(volume > 0,  facecolors=plt.cm.gray(cp.asnumpy(volume))) # Greyscale mapping
ax1.set_title("3D View 1")

print('displaying plot')
plt.tight_layout()
plt.show()