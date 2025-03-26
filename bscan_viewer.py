import cv2
import os 

bscan_dir="d:\\cleaning_GUI_annotated_Data\\Cirrus_OCT_Imaging_Data\\737650964\\R\\20170413\\140118\\OPT\\Carl_Zeiss_Meditec\\200X1024X200\\Original\\B-Scans"

scans=os.listdir(bscan_dir)
print('number of scans:',len(scans))
for i in range(0,len(scans)-4,4):
    hstack=[]
    for j in range(4):
        hstack.append(cv2.imwrite(bscan_dir+os.sep+scans[i+j],0))
    stack=cv2.hconcat(hstack)
    # print('scan')
    # stack=cv2.imread(bscan_dir+os.sep+scans[i],0)
    cv2.imshow('b scans',stack)
    k=cv2.waitKey(0)
    if k==ord('q'):
        break
cv2.destroyAllWindows()