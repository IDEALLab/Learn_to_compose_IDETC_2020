"""
Visualizes data and results

Author(s): Jun Wang (jwang38@umd.edu)
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


def visualize_samples(num, X, Y, save_dir):
    # Visulize Samples
    ind = np.random.choice(X.shape[0], size=num, replace=False)
    inputs = X[ind]
    labels = np.zeros_like(inputs)
    labels[:,:,:,:2] = Y[ind]
    for i in range(num):
        plt.figure()
        plt.subplot(121)
        plt.imshow(inputs[i])
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(labels[i])
        plt.title('Label')
        plt.tight_layout()
        plt.savefig('{}/samples_{}.svg'.format(save_dir, i))
        plt.close()

def visualize(num, X, Y, model, save_dir):
    # Visulize Results
    ind = np.random.choice(X.shape[0], size=num, replace=False)
    inputs = X[ind]
#    predictions = np.zeros_like(inputs)
    predictions = np.zeros((num, 64, 64, 3))
    predictions[:,:,:,:2] = model.predict(inputs)
    labels = np.zeros_like(inputs)
    labels[:,:,:,:2] = Y[ind]
    discrepancies_norm_rel_ave = np.zeros((1,num))
    for i in range(num):
        # Figure size
        fig, axs = plt.subplots(3, 4, figsize=(60,45))
        
        # Input
        inp = axs[0, 1].imshow(inputs[i,:,:,:])
        ind_0 = np.where(inputs[i,:,:,2] == 0)
#        print(inputs[i,:,:,2].shape)
#        fig.colorbar(inp, ax=ax1)
        # axs[0, 1].xaxis.set_tick_params(labelsize=30, colors='grey')
        # axs[0, 1].yaxis.set_tick_params(labelsize=30, colors='grey')
        axs[0, 1].xaxis.set_ticklabels([])
        axs[0, 1].yaxis.set_ticklabels([])
        axs[0, 1].set_title('Input', fontsize=80, color='grey')
        
        # Prediction
        predictions_red = predictions[i, :, :, 0]
        predictions_green = predictions[i, :, :, 1] 
#        print(predictions_red.shape)
        predictions_red[ind_0] = 0
        predictions_green[ind_0] = 0
        predictions[i, :, :, 0] = predictions_red
        predictions[i, :, :, 1] = predictions_green
        predictions_abs = abs(predictions[i])
        predictions_scl = predictions_abs/np.amax(predictions_abs)
        pred = axs[1, 1].imshow(predictions_scl)
#        fig.colorbar(pred, ax=ax2)
        axs[1, 1].xaxis.set_ticklabels([])
        axs[1, 1].yaxis.set_ticklabels([])
        axs[1, 1].set_title('Prediction', fontsize=80, color='grey')
        
        # Scaled Label
        labels_abs = abs(labels[i])
        labels_scl = labels_abs/np.amax(labels_abs)
        lab_scl = axs[1, 2].imshow(labels_scl)
        axs[1, 2].xaxis.set_ticklabels([])
        axs[1, 2].yaxis.set_ticklabels([])
        axs[1, 2].set_title('Ground truth', fontsize=80, color='grey')
        
        # Label norm
        labels_norm = LA.norm(labels[i], axis=2)
        lab_norm = axs[2, 2].imshow(labels_norm, cmap='viridis')
        cbar_lab = fig.colorbar(lab_norm, ax=axs[2, 2], fraction=0.046, pad=0.04)
        cbar_lab.ax.tick_params(labelsize=50, colors='grey')
#        fig.colorbar(lab, ax=ax3)
        axs[2, 2].xaxis.set_ticklabels([])
        axs[2, 2].yaxis.set_ticklabels([])
        axs[2, 2].set_title('Ground truth norm', fontsize=80, color='grey')
        
        # Discrepancy
        discrepancies = predictions[i]-labels[i]
        
        # Discrepancy
        dis_abs = axs[0, 3].imshow(abs(discrepancies))
        axs[0, 3].xaxis.set_ticklabels([])
        axs[0, 3].yaxis.set_ticklabels([])
        axs[0, 3].set_title('Absolute discrepancy ', fontsize=80, color='grey')
        
        # Scaled Discrepancy
        discrepancies_abs = abs(discrepancies)
        discrepancies_scl = discrepancies_abs/np.amax(discrepancies_abs)
        dis_scl = axs[1, 3].imshow(discrepancies_scl)
        axs[1, 3].xaxis.set_ticklabels([])
        axs[1, 3].yaxis.set_ticklabels([])
        axs[1, 3].set_title('Scaled discrepancy ', fontsize=80, color='grey')
        
        # Discrepancy norm
        discrepancies_norm = LA.norm(discrepancies, axis=2)
        discrepancies_norm_rel = discrepancies_norm / 1
        dis_norm = axs[2, 3].imshow(discrepancies_norm_rel, cmap='viridis')
        cbar_dis = fig.colorbar(dis_norm, ax=axs[2, 3], fraction=0.046, pad=0.04)
        cbar_dis.ax.tick_params(labelsize=50, colors='grey')
        axs[2, 3].xaxis.set_ticklabels([])
        axs[2, 3].yaxis.set_ticklabels([])
        axs[2, 3].set_title('Discrepancy norm', fontsize=80, color='grey')
        
        # Relative norm
        # discrepancies_norm = LA.norm(discrepancies, axis=2)
        predictions_norm = LA.norm(predictions[i], axis=2)
        # discrepancies_norm_rel = discrepancies_norm / (np.maximum(labels_norm, LA.norm(predictions[i], axis=2)))
        PRE_norm = abs(predictions_norm-labels_norm)
        N_valid = np.count_nonzero(labels_norm)
        labels_norm[labels_norm==0]=math.inf
        # discrepancies_norm_rel = abs(discrepancies_norm) / labels_norm
        discrepancies_norm_rel = PRE_norm / labels_norm
        discrepancies_norm_rel_ave[:,i] = np.sum(discrepancies_norm_rel)/N_valid
        print(N_valid)
        # print(discrepancies_norm_rel_ave)
        discrepancies_norm_rel[discrepancies_norm_rel>=0.06]=0
        dis_norm = axs[2, 1].imshow(discrepancies_norm_rel, cmap='viridis')
        cbar_dis = fig.colorbar(dis_norm, ax=axs[2, 1], fraction=0.046, pad=0.04)
        cbar_dis.ax.tick_params(labelsize=50, colors='grey')
        axs[2, 1].xaxis.set_ticklabels([])
        axs[2, 1].yaxis.set_ticklabels([])
        axs[2, 1].set_title('Relative norm', fontsize=80, color='grey')
        
        # Hide the axes of unused subplots
        axs[0, 0].axis('off')
        axs[1, 0].axis('off')
        axs[0, 2].axis('off')
        axs[2, 0].axis('off')
        # axs[2, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('{}/{}_{}.svg'.format(save_dir, model.name, i))
        plt.close()
    
    print(np.sum(discrepancies_norm_rel_ave)/num)
        









