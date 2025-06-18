import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import os


def rocPlotter(y_score,y_true,n_classes,save_dir,save_metrics=True,classNames=None):
    
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
        key=classNames[key] if classNames is not None else i
        fpr[key], tpr[key], thresholds[key] = roc_curve(y_true[:, i], y_score[:, i])
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

    if save_metrics:
        with open(save_dir+os.sep+"eval_matrix.json",'a') as f:
            json.dump({
                'fpr':fpr,
                'tpr':tpr,
                'threshold':thresholds,
                'roc_auc':roc_auc,
                'optimal threshold':optimal_thresholds,
            },f,indent=4)

        plt.savefig(save_dir)
        
    else:
        plt.show()
    
    return optimal_thresholds