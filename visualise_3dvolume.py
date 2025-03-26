import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import cv2
from tqdm import tqdm
# Create sample 3D volume
# volume = np.random.randint(0, 255, (50, 50, 50), dtype=np.uint8)
bscan_dir=r"d:\cleaning_GUI_annotated_Data\Cirrus_OCT_Imaging_Data\000003162\L\20080818\124239\OPT\Carl_Zeiss_Meditec\200X1024X200\Original\B-Scans"
images=sorted(os.listdir(bscan_dir),key=lambda x:int(x.split("_")[-1].split('.')[0].strip()))
# print(images)
volume=[]
# for i,image in tqdm(enumerate(images)):
for i in range(0,len(images),4):
    # if i>5:
    #     break
    image_path=os.path.join(bscan_dir+os.sep+images[i])
    image=cv2.imread(image_path,0)
    image=cv2.resize(image,(image.shape[0]//4,image.shape[1]//4))
    # print(image.shape)
    cv2.imshow('bscan',image)
    cv2.waitKey(1)
    volume.append(image.tolist())
volume=np.array(volume)/255
# print('unique values of volume :',np.unique(volume))
print('the shape of volume is:',volume.shape)
x, y, z = np.meshgrid(np.linspace(0, volume.shape[0], volume.shape[0]),
                      np.linspace(0, volume.shape[1], volume.shape[1]),
                      np.linspace(0, volume.shape[2], volume.shape[2]))
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot isosurface
ax.scatter(x, y, z,c=volume, cmap='binary', alpha=1)

# Enable interactive rotation
ax.view_init(elev=30, azim=45)  # Initial viewing angles
plt.show()
