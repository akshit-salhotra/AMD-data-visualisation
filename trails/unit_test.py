import torch

def getTensor(amdtype):
    classes={'early':0,'inter':1,'ga':2,'wet':3,'scar':4,'notAMD':5}
    label=torch.zeros(len(classes))
    for c in amdtype:
        label[classes[c]]=1

    print(label)


if __name__=="__main__":
        classes={'early':0,'inter':1,'ga':2,'wet':3,'scar':4,'notAMD':5}
        for keys in classes.keys():
            getTensor([keys])
        getTensor(['early','notAMD'])
        getTensor(['ga','wet'])
        getTensor(['wet','scar'])