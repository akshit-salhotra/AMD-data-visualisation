import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,confusion_matrix
import json
import os
from itertools import combinations

def rocPlotter(y_score,y_true,n_classes,save_dir,json_dir,save_metrics=True,classNames=None):
    
    # Initialize metrics storage
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    optimal_thresholds = dict()
    
    if classNames is not None:
        assert len(classNames)==n_classes,f'the classNames must have same length as number of classes \n n_classes:{n_classes} \n length of classNames:{len(classNames)}'
    
    # Compute ROC and optimal thresholds
    for i in range(n_classes):
        key=classNames[i] if classNames is not None else i
        fpr[key], tpr[key], thresholds[key] = roc_curve(y_true[:, i].squeeze(), y_score[:, i].squeeze())
        roc_auc[key] = auc(fpr[key], tpr[key])
        J_scores = tpr[key] - fpr[key]
        ix = np.argmax(J_scores)
        optimal_thresholds[key] = thresholds[key][ix]
        if classNames is not None:
            print(f'Class {classNames[i]}: Optimal threshold = {optimal_thresholds[key]:.2f}, AUC = {roc_auc[key]:.2f}')
        else:
            print(f'Class {i}: Optimal threshold = {optimal_thresholds[key]:.2f}, AUC = {roc_auc[key]:.2f}')


    # Plotting ROC curves in 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.ravel()  # flatten the 2D array of axes

    for i in range(n_classes):
        key=classNames[i] if classNames is not None else i
       
        axs[i].plot(fpr[key], tpr[key], label=f'AUC = {roc_auc[key]:.2f}', color='C0')
        
        # Mark optimal threshold
        optimal_idx = np.argmax(tpr[key] - fpr[key])
        axs[i].plot(fpr[key][optimal_idx], tpr[key][optimal_idx], 'ro')
        axs[i].text(fpr[key][optimal_idx], tpr[key][optimal_idx], 
                    f'Th={optimal_thresholds[key]:.2f}', fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right')
        
        axs[i].plot([0, 1], [0, 1], 'k--', lw=1)
        axs[i].set_xlim([0.0, 1.0])
        axs[i].set_ylim([0.0, 1.05])
        axs[i].set_title(f'ROC Curve - Class {key}')
        axs[i].set_xlabel('FPR')
        axs[i].set_ylabel('TPR')
        axs[i].legend(loc='lower right')
        axs[i].grid(True)
        plt.tight_layout()

    for key in tpr.keys():
    #     tpr[key]=tpr[key].tolist()
    #     fpr[key]=fpr[key].tolist()
    #     for i,val in enumerate(thresholds[key]):
    #         if np.isnan(val):
    #             thresholds[key][i]=1
    #     thresholds[key]=thresholds[key].tolist()
        roc_auc[key]=str(roc_auc[key])
        optimal_thresholds[key]=str(optimal_thresholds[key]) if not np.isnan(optimal_thresholds[key]) else "1" 

    if save_metrics:
        with open(json_dir+os.sep+"eval_matrix.json",'w') as f:
            print(roc_auc,optimal_thresholds)
            json.dump({
                # 'fpr':fpr,
                # 'tpr':tpr,
                # 'threshold':thresholds,
                'roc_auc':roc_auc,
                'optimal threshold':optimal_thresholds,
            },f,indent=4)

        plt.savefig(save_dir)
        
    else:
        plt.show()
    print('returing metrics!!!')
    return optimal_thresholds

def multilabel_confusionMatrixSoftmax(logits:np.ndarray,labels:np.ndarray,classes:dict,multiLabel:list):
    
    '''
    Requires:
    
    logits:(n,n_classes)
    labels:(n,n_classes)
    multiLabel:(n_classes,)
    
    returns:
    
    cm :sklearn.mertics.confusion_matirix
    
    make sure the order of classes is sorted in multiLabel list !!!!
     
    '''
    
    preds=np.argmax(logits,axis=-1)
    
    labels_classes=[[key[0]] for key in sorted(classes.items(),key=lambda i:i[1])]
    print(labels_classes)
    comb=[]
    
    multiLabel_classes=[labels_classes[i] for i in range(len(multiLabel)) if multiLabel[i]==1]
    
    multiLabel_encoded=[]
    
    print(multiLabel_classes)


    
    
    for i in range(2,len(multiLabel)+1):
        comb.extend(combinations(multiLabel_classes,i))
    
    print(comb)
    for c in comb:
        encoded=np.zeros(len(multiLabel))
        for i in c:
            encoded[classes[i[0]]]=1
        multiLabel_encoded.append(encoded)

    labels_classes=labels_classes+comb
    
    y_true=[]
    one_hot_encoded=[]
    for i in range(len(multiLabel)):
        one_hot_encoded.append([1 if idx==i else 0 for idx in range(len(multiLabel))])
    multiLabel_encoded.extend(one_hot_encoded)
    print(multiLabel_encoded)
    print(labels)
    for label in labels:
        for i,ls in enumerate(multiLabel_encoded):
            if (ls==label).all():
                y_true.append(i)
                break
    
    y_true=np.array(y_true)
    
    print(y_true.shape,preds.shape)
    labels_classes=[str(v) for v in labels_classes]
    cm=confusion_matrix(y_true,preds,labels=[i for i in range(len(labels_classes))])
    
    return cm,labels_classes
        
    
    
    
if __name__=="__main__":
    # import torch
    # a=torch.bernoulli(torch.full((305, 6), 0.5))
    # b=torch.bernoulli(torch.full((305, 6), 0.5))

    # rocPlotter(a,b,6,"D:/AMD-data-visualisation")
    # logits=np.random.randint(-100,100,(100,6))
    # y_true=np.random.randint(0,2,(100,6))
    # print(np.unique(y_true))
    # d={'Early AMD':0,'Int AMD':1,'GA':2,'Wet':3,'Scar':4,"Not AMD":5}
    m=[0,0,1,1,1,0]

    # print(multilabel_confusionMatrixSoftmax(logits,y_true,d,m))