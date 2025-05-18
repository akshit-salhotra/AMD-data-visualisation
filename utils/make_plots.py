import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def get_batch_stats_plot(batchnorm_stats,save_path):
    for name in batchnorm_stats['val'].keys():
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'BatchNorm Stats (Train vs Val) - {name}', fontsize=20)

        # Prepare
        train_stats = batchnorm_stats['train'][name]
        val_stats = batchnorm_stats['val'][name]

        train_batch_mean = torch.stack(train_stats["batch_mean"]).numpy()
        train_running_mean = torch.stack(train_stats["running_mean"]).numpy()
        train_batch_std = torch.stack(train_stats["batch_std"]).numpy()
        train_running_std = torch.stack(train_stats["running_std"]).numpy()

        val_batch_mean = torch.stack(val_stats["batch_mean"]).numpy()
        val_running_mean = torch.stack(val_stats["running_mean"]).numpy()
        val_batch_std = torch.stack(val_stats["batch_std"]).numpy()
        val_running_std = torch.stack(val_stats["running_std"]).numpy()

        # Train mean
        axs[0, 0].plot(train_batch_mean.mean(axis=1), label='Train Batch Mean')
        axs[0, 0].plot(train_running_mean.mean(axis=1), label='Train Running Mean')
        axs[0, 0].set_title('Train Mean')
        axs[0, 0].legend()

        # Train std
        axs[0, 1].plot(train_batch_std.mean(axis=1), label='Train Batch Std')
        axs[0, 1].plot(train_running_std.mean(axis=1), label='Train Running Std')
        axs[0, 1].set_title('Train Std')
        axs[0, 1].legend()

        # Val mean
        axs[1, 0].plot(val_batch_mean.mean(axis=1), label='Val Batch Mean')
        axs[1, 0].plot(val_running_mean.mean(axis=1), label='Val Running Mean')
        axs[1, 0].set_title('Val Mean')
        axs[1, 0].legend()

        # Val std
        axs[1, 1].plot(val_batch_std.mean(axis=1), label='Val Batch Std')
        axs[1, 1].plot(val_running_std.mean(axis=1), label='Val Running Std')
        axs[1, 1].set_title('Val Std')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path+name+".png")
        plt.close()
        # print('saved stats')