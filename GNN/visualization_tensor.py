"""
Visualize data and results of GNN (tensor version)

Author: Jun Wang (jwang38@umd.edu)
"""
import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def visualize(num, X_first, Y_first, X_next, Y_next, edge_feature, model, save_dir):  
    '''
    Visualize the results by comparing the composed input and output graphs
    to the composed target graphs.

    Args:
    num : Number of output graphs to visualize.
    X_first : The first pipe of the input graph for visualizing the composition.
    Y_first : The first pipe of the target graph for visualizing the composition.
    X_next : The second pipe of the input graph for visualizing the composition.
    Y_next : The second pipe of the target graph for visualizing the composition.
    edge_feature : The edge fature to distinguish between horizontal connections
                   and vertical connections.
    model : The trained or restored GNN model.
    save_dir : Directory of saving the visualization results.
    '''
    # Randomly select certain number of evaluation data for visualization
    ind = np.random.choice(len(X_first), size = num, replace=False)
    # ind = np.random.choice(num, size = num, replace=False)
    ind.sort()
    print(ind)
    # inputs = [X_graph[i] for i in ind]
    
    results = model.predict(X_first, X_next, edge_feature)
    predicted_two_node_graph_tes = np.split(results, len(results)/2, axis=0)
    
    # Edge
    edges = np.zeros((num, 1))
    for i in range(len(ind)):
        edges[i] = edge_feature[ind[i], ...]
    # Input
    inputs_hor = np.zeros((num, 64, 128, 3))
    inputs_ver = np.zeros((num, 128, 64, 3))
    inputs_verR = np.zeros((num, 128, 64, 3))
    for i in range(len(ind)): 
        inputs_hor[i,:,:64,:2] = X_first[ind[i], ...]
        inputs_hor[i,:,64:,:2] = X_next[ind[i], ...]
        inputs_ver[i,64:,:,:2] = X_first[ind[i], ...]
        inputs_ver[i,:64,:,:2] = X_next[ind[i], ...]
        inputs_verR[i,:64,:,:2] = X_first[ind[i], ...]
        inputs_verR[i,64:,:,:2] = X_next[ind[i], ...]
    # Target
    targets_hor = np.zeros((num, 64, 128, 3))
    targets_ver = np.zeros((num, 128, 64, 3))
    targets_verR = np.zeros((num, 128, 64, 3))
    for i in range(len(ind)):
        targets_hor[i,:,:64,:2] = Y_first[ind[i], ...]
        targets_hor[i,:,64:,:2] = Y_next[ind[i], ...]
        targets_ver[i,64:,:,:2] = Y_first[ind[i], ...]
        targets_ver[i,:64,:,:2] = Y_next[ind[i], ...]
        targets_verR[i,:64,:,:2] = Y_first[ind[i], ...]
        targets_verR[i,64:,:,:2] = Y_next[ind[i], ...]
    # Prediction
    predictions_hor = np.zeros((num, 64, 128, 3))
    predictions_ver = np.zeros((num, 128, 64, 3))
    predictions_verR = np.zeros((num, 128, 64, 3))
    for i in range(len(ind)):
        predictions_hor[i,:,:64,:2] = np.reshape(predicted_two_node_graph_tes[ind[i]][0, :], (64, 64, 2))
        predictions_hor[i,:,64:,:2] = np.reshape(predicted_two_node_graph_tes[ind[i]][1, :], (64, 64, 2))
        predictions_ver[i,64:,:,:2] = np.reshape(predicted_two_node_graph_tes[ind[i]][0, :], (64, 64, 2))
        predictions_ver[i,:64,:,:2] = np.reshape(predicted_two_node_graph_tes[ind[i]][1, :], (64, 64, 2))
        predictions_verR[i,:64,:,:2] = np.reshape(predicted_two_node_graph_tes[ind[i]][0, :], (64, 64, 2))
        predictions_verR[i,64:,:,:2] = np.reshape(predicted_two_node_graph_tes[ind[i]][1, :], (64, 64, 2))
    
    # Plot
    discrepancies_norm_rel_ave = np.zeros((1,num))
    inp_discrepancies_norm_rel_ave = np.zeros((1,num))
    for i in range(num):
        #figure size
        fig, axs = plt.subplots(3, 4, figsize=(60,45), constrained_layout=True)
        
        if edges[i] == 0: # Horizontal connections
            # Shape index for non-pipe area
            ind_0 = np.where(targets_hor[i,:,:,0] == 0)
            
            # Input
            # Make the non-pipe area zeros
            inputs_hor[i,:,:,0][ind_0] = 0
            inputs_hor[i,:,:,1][ind_0] = 0
            inp = axs[0, 0].imshow(abs(inputs_hor[i]))
            # axs[0, 0].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 0].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 0].get_xaxis().set_visible(False)
            axs[0, 0].get_yaxis().set_visible(False)
            axs[0, 0].set_title('Input composition', fontsize=60, color='grey') 
            
            # Target
            tar = axs[0, 1].imshow(abs(targets_hor[i]))
            # axs[0, 1].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 1].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 1].get_xaxis().set_visible(False)
            axs[0, 1].get_yaxis().set_visible(False)
            axs[0, 1].set_title('Ground truth composition', fontsize=60, color='grey')
            
            # Discrepancy (Input vs. Target)
            discrepancies_inp = inputs_hor[i]-targets_hor[i]
            
            # Absolute Discrepancy (Input vs. Target)
            dis = axs[0, 2].imshow(abs(discrepancies_inp*10))
            # axs[0, 2].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 2].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 2].get_xaxis().set_visible(False)
            axs[0, 2].get_yaxis().set_visible(False)
            axs[0, 2].set_title('Naive composition discrepancy', fontsize=60, color='grey')
            
            # Discrepancy norm (Input vs. Target)
            discrepancies_inp_norm = LA.norm(discrepancies_inp, axis=2)
            discrepancies_inp_norm_rel = discrepancies_inp_norm / 1
            dis_norms = axs[0, 3].imshow(discrepancies_inp_norm_rel, cmap='viridis')
            # cbar_dis_inp = fig.colorbar(dis_inp_norm, ax=axs[0, 3], fraction=0.046, pad=0.04)
            # cbar_dis_inp.ax.tick_params(labelsize=50, colors='grey')
            # axs[0, 3].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 3].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 3].get_xaxis().set_visible(False)
            axs[0, 3].get_yaxis().set_visible(False)
            axs[0, 3].set_title('Naive discrepancy norm', fontsize=60, color='grey')
            
            # Prediction
            predictions_hor[i,:,:,0][ind_0] = 0
            predictions_hor[i,:,:,1][ind_0] = 0
            pred = axs[1, 0].imshow(abs(predictions_hor[i]))
            # axs[1, 0].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 0].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 0].get_xaxis().set_visible(False)
            axs[1, 0].get_yaxis().set_visible(False)
            axs[1, 0].set_title('Predicted composition', fontsize=60, color='grey')
            
            # Target
            tar = axs[1, 1].imshow(abs(targets_hor[i]))
            # axs[1, 1].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 1].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 1].get_xaxis().set_visible(False)
            axs[1, 1].get_yaxis().set_visible(False)
            axs[1, 1].set_title('Ground truth composition', fontsize=60, color='grey')
            
            # Target norm
            targets_norm = LA.norm(targets_hor[i], axis=2)
            tar_norm = axs[2, 1].imshow(targets_norm, cmap='viridis')
            cbar_tar = fig.colorbar(tar_norm, ax=axs[2, 1], fraction=0.046, pad=0.04)
            cbar_tar.ax.tick_params(labelsize=50, colors='grey')
            # axs[2, 1].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[2, 1].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[2, 1].get_xaxis().set_visible(False)
            axs[2, 1].get_yaxis().set_visible(False)
            axs[2, 1].set_title('Ground truth norm', fontsize=60, color='grey')
            
            # Discrepancy (Prediction vs. Target)
            discrepancies_pred = predictions_hor[i]-targets_hor[i]
    
            # Absolute Discrepancy (Prediction vs. Target)
            dis = axs[1, 2].imshow(abs(discrepancies_pred*10))
            # axs[1, 2].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 2].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 2].get_xaxis().set_visible(False)
            axs[1, 2].get_yaxis().set_visible(False)
            axs[1, 2].set_title('Predicted composition discrepancy', fontsize=60, color='grey')
            
            # Discrepancy norm (Prediction vs. Target)
            discrepancies_pred_norm = LA.norm(discrepancies_pred, axis=2)
            discrepancies_pred_norm_rel = discrepancies_pred_norm / 1
            discrepancies_pred_norm_ave = np.sum(discrepancies_pred_norm_rel)/1344/2
            # print(discrepancies_pred_norm_ave)
            discrepancies_pred_norm_var = discrepancies_pred_norm_rel-discrepancies_pred_norm_ave
            discrepancies_pred_norm_var[discrepancies_pred_norm_var==-discrepancies_pred_norm_ave]=0
            discrepancies_pred_norm_dev = np.sqrt(np.sum(np.square(discrepancies_pred_norm_var))/1344/2)
            # print(discrepancies_pred_norm_dev)
            dis_norms = axs[1, 3].imshow(discrepancies_pred_norm_rel, cmap='viridis')
            cbar_dis_pred = fig.colorbar(dis_norms, ax=axs[0:2, 3], shrink=0.5, aspect=10, fraction=0.046, pad=0.04)
            cbar_dis_pred.ax.tick_params(labelsize=50, colors='grey')
            # axs[1, 3].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 3].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 3].get_xaxis().set_visible(False)
            axs[1, 3].get_yaxis().set_visible(False)
            axs[1, 3].set_title('Predicted discrepancy norm', fontsize=60, color='grey')
            
            # Relative norms
            inputs_norm = LA.norm(inputs_hor[i], axis=2)
            inputs_PRE_norm = abs(inputs_norm-targets_norm)
            
            predictions_norm = LA.norm(predictions_hor[i], axis=2)
            predictions_PRE_norm = abs(predictions_norm-targets_norm)
            
            N_valid = np.count_nonzero(targets_norm)
            targets_norm[targets_norm==0]=math.inf
            
            # Input Relative norm
            inp_discrepancies_norm_rel = inputs_PRE_norm / targets_norm
            inp_discrepancies_norm_rel_ave[:,i] = np.sum(inp_discrepancies_norm_rel)/N_valid
            print('input average PRE:' + str(inp_discrepancies_norm_rel_ave))
            inp_discrepancies_norm_rel[inp_discrepancies_norm_rel>=0.3]=0
            norms = axs[2, 2].imshow(inp_discrepancies_norm_rel, cmap='viridis')
            # axs[2, 2].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[2, 2].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[2, 2].get_xaxis().set_visible(False)
            axs[2, 2].get_yaxis().set_visible(False)
            axs[2, 2].set_title('Input Relative norm', fontsize=60, color='grey')
            
            # Prediction Relative norm
            # discrepancies_norm_rel = discrepancies_pred_norm / targets_norm
            discrepancies_norm_rel = predictions_PRE_norm/targets_norm
            discrepancies_norm_rel_ave[:,i] = np.sum(discrepancies_norm_rel)/N_valid
            print(N_valid)
            print('prediction average PRE:' + str(discrepancies_norm_rel_ave))
            discrepancies_norm_rel[discrepancies_norm_rel>=0.3]=0
            norms = axs[2, 3].imshow(discrepancies_norm_rel, cmap='viridis')
            cbar_dis = fig.colorbar(norms, ax=axs[2, 2:4], fraction=0.046, pad=0.04)
            cbar_dis.ax.tick_params(labelsize=50, colors='grey')
            # axs[2, 3].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[2, 3].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[2, 3].get_xaxis().set_visible(False)
            axs[2, 3].get_yaxis().set_visible(False)
            axs[2, 3].set_title('Predicted Relative norm', fontsize=60, color='grey')
            
        elif edges[i] == 1: # Vertical connections
            # Shape index for non-pipe area
            ind_0 = np.where(targets_ver[i,:,:,0] == 0)
            
            # Input
            # Make the non-pipe area zeros
            inputs_ver[i,:,:,0][ind_0] = 0
            inputs_ver[i,:,:,1][ind_0] = 0
            inp = axs[0, 0].imshow(abs(inputs_ver[i]))
            # axs[0, 0].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 0].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 0].get_xaxis().set_visible(False)
            axs[0, 0].get_yaxis().set_visible(False)
            axs[0, 0].set_title('Input composition', fontsize=60, color='grey') 
            
            # Target
            tar = axs[0, 1].imshow(abs(targets_ver[i]))
            # axs[0, 1].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 1].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 1].get_xaxis().set_visible(False)
            axs[0, 1].get_yaxis().set_visible(False)
            axs[0, 1].set_title('Ground truth composition', fontsize=60, color='grey')
            
            # Discrepancy (Input vs. Target)
            discrepancies_inp = inputs_ver[i]-targets_ver[i]
            
            # Absolute Discrepancy (Input vs. Target)
            dis = axs[0, 2].imshow(abs(discrepancies_inp*10))
            # axs[0, 2].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 2].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 2].get_xaxis().set_visible(False)
            axs[0, 2].get_yaxis().set_visible(False)
            axs[0, 2].set_title('Naive composition discrepancy', fontsize=60, color='grey')
            
            # Discrepancy norm (Input vs. Target)
            discrepancies_inp_norm = LA.norm(discrepancies_inp, axis=2)
            discrepancies_inp_norm_rel = discrepancies_inp_norm / 1
            dis_norms = axs[0, 3].imshow(discrepancies_inp_norm_rel, cmap='viridis')
            # cbar_dis_inp = fig.colorbar(dis_inp_norm, ax=axs[0, 3], fraction=0.046, pad=0.04)
            # cbar_dis_inp.ax.tick_params(labelsize=50, colors='grey')
            # axs[0, 3].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 3].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 3].get_xaxis().set_visible(False)
            axs[0, 3].get_yaxis().set_visible(False)
            axs[0, 3].set_title('Naive discrepancy norm', fontsize=60, color='grey')
            
            # Prediction
            predictions_ver[i,:,:,0][ind_0] = 0
            predictions_ver[i,:,:,1][ind_0] = 0
            pred = axs[1, 0].imshow(abs(predictions_ver[i]))
            # axs[1, 0].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 0].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 0].get_xaxis().set_visible(False)
            axs[1, 0].get_yaxis().set_visible(False)
            axs[1, 0].set_title('Predicted composition', fontsize=60, color='grey')
            
            # Target
            tar = axs[1, 1].imshow(abs(targets_ver[i]))
            # axs[1, 1].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 1].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 1].get_xaxis().set_visible(False)
            axs[1, 1].get_yaxis().set_visible(False)
            axs[1, 1].set_title('Ground truth composition', fontsize=60, color='grey')
            
            # Target norm
            targets_norm = LA.norm(targets_ver[i], axis=2)
            tar_norm = axs[2, 1].imshow(targets_norm, cmap='viridis')
            cbar_tar = fig.colorbar(tar_norm, ax=axs[2, 1], fraction=0.046, pad=0.04)
            cbar_tar.ax.tick_params(labelsize=50, colors='grey')
    #        fig.colorbar(lab, ax=ax3)
            # axs[2, 1].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[2, 1].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[2, 1].get_xaxis().set_visible(False)
            axs[2, 1].get_yaxis().set_visible(False)
            axs[2, 1].set_title('Ground truth norm', fontsize=60, color='grey')
            
            # Discrepancy (Prediction vs. Target)
            discrepancies_pred = predictions_ver[i]-targets_ver[i]
    
            # Absolute Discrepancy (Prediction vs. Target)
            dis = axs[1, 2].imshow(abs(discrepancies_pred*10))
            # axs[1, 2].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 2].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 2].get_xaxis().set_visible(False)
            axs[1, 2].get_yaxis().set_visible(False)
            axs[1, 2].set_title('Predicted composition discrepancy', fontsize=60, color='grey')
            
            # Discrepancy norm (Prediction vs. Target)
            discrepancies_pred_norm = LA.norm(discrepancies_pred, axis=2)
            discrepancies_pred_norm_rel = discrepancies_pred_norm / 1
            discrepancies_pred_norm_ave = np.sum(discrepancies_pred_norm_rel)/1344/2
            # print(discrepancies_pred_norm_ave)
            discrepancies_pred_norm_var = discrepancies_pred_norm_rel-discrepancies_pred_norm_ave
            discrepancies_pred_norm_var[discrepancies_pred_norm_var==-discrepancies_pred_norm_ave]=0
            discrepancies_pred_norm_dev = np.sqrt(np.sum(np.square(discrepancies_pred_norm_var))/1344/2)
            # print(discrepancies_pred_norm_dev)
            dis_norms = axs[1, 3].imshow(discrepancies_pred_norm_rel, cmap='viridis')
            cbar_dis_pred = fig.colorbar(dis_norms, ax=axs[0:2, 3], shrink=0.5, aspect=10, fraction=0.046, pad=0.04)
            cbar_dis_pred.ax.tick_params(labelsize=50, colors='grey')
            # axs[1, 3].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 3].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 3].get_xaxis().set_visible(False)
            axs[1, 3].get_yaxis().set_visible(False)
            axs[1, 3].set_title('Predicted discrepancy norm', fontsize=60, color='grey')
            
            # Relative norms
            inputs_norm = LA.norm(inputs_ver[i], axis=2)
            inputs_PRE_norm = abs(inputs_norm-targets_norm)
            
            predictions_norm = LA.norm(predictions_ver[i], axis=2)
            predictions_PRE_norm = abs(predictions_norm-targets_norm)
            
            N_valid = np.count_nonzero(targets_norm)
            targets_norm[targets_norm==0]=math.inf
            
            # Input Relative norm
            # inp_discrepancies_norm_rel = discrepancies_inp_norm / targets_norm
            inp_discrepancies_norm_rel = inputs_PRE_norm / targets_norm
            inp_discrepancies_norm_rel_ave[:,i] = np.sum(inp_discrepancies_norm_rel)/N_valid
            print('input average PRE:' + str(inp_discrepancies_norm_rel_ave))
            inp_discrepancies_norm_rel[inp_discrepancies_norm_rel>=0.3]=0
            norms = axs[2, 2].imshow(inp_discrepancies_norm_rel, cmap='viridis')
            # axs[2, 2].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[2, 2].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[2, 2].get_xaxis().set_visible(False)
            axs[2, 2].get_yaxis().set_visible(False)
            axs[2, 2].set_title('Input Relative norm', fontsize=60, color='grey')
            
            # Prediction Relative norm
            # discrepancies_norm_rel = discrepancies_pred_norm / targets_norm
            discrepancies_norm_rel = predictions_PRE_norm/targets_norm
            discrepancies_norm_rel_ave[:,i] = np.sum(discrepancies_norm_rel)/N_valid
            print(N_valid)
            print('prediction average PRE:' + str(discrepancies_norm_rel_ave))
            discrepancies_norm_rel[discrepancies_norm_rel>=0.3]=0
            norms = axs[2, 3].imshow(discrepancies_norm_rel, cmap='viridis')
            cbar_dis = fig.colorbar(norms, ax=axs[2, 2:4], fraction=0.046, pad=0.04)
            cbar_dis.ax.tick_params(labelsize=50, colors='grey')
            # axs[2, 3].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[2, 3].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[2, 3].get_xaxis().set_visible(False)
            axs[2, 3].get_yaxis().set_visible(False)
            axs[2, 3].set_title('Predicted Relative norm', fontsize=60, color='grey')
            
        elif edges[i] == -1: # Vertical connections
            # Shape index for non-pipe area
            ind_0 = np.where(targets_verR[i,:,:,0] == 0)
            
            # Input
            # Make the non-pipe area zeros
            inputs_verR[i,:,:,0][ind_0] = 0
            inputs_verR[i,:,:,1][ind_0] = 0
            inp = axs[0, 0].imshow(abs(inputs_verR[i]))
            # axs[0, 0].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 0].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 0].get_xaxis().set_visible(False)
            axs[0, 0].get_yaxis().set_visible(False)
            axs[0, 0].set_title('Input composition', fontsize=60, color='grey') 
            
            # Target
            tar = axs[0, 1].imshow(abs(targets_verR[i]))
            # axs[0, 1].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 1].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 1].get_xaxis().set_visible(False)
            axs[0, 1].get_yaxis().set_visible(False)
            axs[0, 1].set_title('Ground truth composition', fontsize=60, color='grey')
            
            # Discrepancy (Input vs. Target)
            discrepancies_inp = inputs_verR[i]-targets_verR[i]
            
            # Absolute Discrepancy (Input vs. Target)
            dis = axs[0, 2].imshow(abs(discrepancies_inp*10))
            # axs[0, 2].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 2].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 2].get_xaxis().set_visible(False)
            axs[0, 2].get_yaxis().set_visible(False)
            axs[0, 2].set_title('Naive composition discrepancy', fontsize=60, color='grey')
            
            # Discrepancy norm (Input vs. Target)
            discrepancies_inp_norm = LA.norm(discrepancies_inp, axis=2)
            discrepancies_inp_norm_rel = discrepancies_inp_norm / 1
            dis_norms = axs[0, 3].imshow(discrepancies_inp_norm_rel, cmap='viridis')
            # cbar_dis_inp = fig.colorbar(dis_inp_norm, ax=axs[0, 3], fraction=0.046, pad=0.04)
            # cbar_dis_inp.ax.tick_params(labelsize=50, colors='grey')
            # axs[0, 3].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[0, 3].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[0, 3].get_xaxis().set_visible(False)
            axs[0, 3].get_yaxis().set_visible(False)
            axs[0, 3].set_title('Naive discrepancy norm', fontsize=60, color='grey')
            
            # Prediction
            predictions_verR[i,:,:,0][ind_0] = 0
            predictions_verR[i,:,:,1][ind_0] = 0
            pred = axs[1, 0].imshow(abs(predictions_verR[i]))
            # axs[1, 0].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 0].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 0].get_xaxis().set_visible(False)
            axs[1, 0].get_yaxis().set_visible(False)
            axs[1, 0].set_title('Predicted composition', fontsize=60, color='grey')
            
            # Target
            tar = axs[1, 1].imshow(abs(targets_verR[i]))
            # axs[1, 1].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 1].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 1].get_xaxis().set_visible(False)
            axs[1, 1].get_yaxis().set_visible(False)
            axs[1, 1].set_title('Ground truth composition', fontsize=60, color='grey')
            
            # Target norm
            targets_norm = LA.norm(targets_verR[i], axis=2)
            tar_norm = axs[2, 1].imshow(targets_norm, cmap='viridis')
            cbar_tar = fig.colorbar(tar_norm, ax=axs[2, 1], fraction=0.046, pad=0.04)
            cbar_tar.ax.tick_params(labelsize=50, colors='grey')
    #        fig.colorbar(lab, ax=ax3)
            # axs[2, 1].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[2, 1].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[2, 1].get_xaxis().set_visible(False)
            axs[2, 1].get_yaxis().set_visible(False)
            axs[2, 1].set_title('Ground truth norm', fontsize=60, color='grey')
            
            # Discrepancy (Prediction vs. Target)
            discrepancies_pred = predictions_verR[i]-targets_verR[i]
    
            # Absolute Discrepancy (Prediction vs. Target)
            dis = axs[1, 2].imshow(abs(discrepancies_pred*10))
            # axs[1, 2].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 2].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 2].get_xaxis().set_visible(False)
            axs[1, 2].get_yaxis().set_visible(False)
            axs[1, 2].set_title('Predicted composition discrepancy', fontsize=60, color='grey')
            
            # Discrepancy norm (Prediction vs. Target)
            discrepancies_pred_norm = LA.norm(discrepancies_pred, axis=2)
            discrepancies_pred_norm_rel = discrepancies_pred_norm / 1
            discrepancies_pred_norm_ave = np.sum(discrepancies_pred_norm_rel)/1344/2
            # print(discrepancies_pred_norm_ave)
            discrepancies_pred_norm_var = discrepancies_pred_norm_rel-discrepancies_pred_norm_ave
            discrepancies_pred_norm_var[discrepancies_pred_norm_var==-discrepancies_pred_norm_ave]=0
            discrepancies_pred_norm_dev = np.sqrt(np.sum(np.square(discrepancies_pred_norm_var))/1344/2)
            # print(discrepancies_pred_norm_dev)
            dis_norms = axs[1, 3].imshow(discrepancies_pred_norm_rel, cmap='viridis')
            cbar_dis_pred = fig.colorbar(dis_norms, ax=axs[0:2, 3], shrink=0.5, aspect=10, fraction=0.046, pad=0.04)
            cbar_dis_pred.ax.tick_params(labelsize=50, colors='grey')
            # axs[1, 3].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[1, 3].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[1, 3].get_xaxis().set_visible(False)
            axs[1, 3].get_yaxis().set_visible(False)
            axs[1, 3].set_title('Predicted discrepancy norm', fontsize=60, color='grey')
            
            # Relative norms
            inputs_norm = LA.norm(inputs_verR[i], axis=2)
            inputs_PRE_norm = abs(inputs_norm-targets_norm)
            
            predictions_norm = LA.norm(predictions_verR[i], axis=2)
            predictions_PRE_norm = abs(predictions_norm-targets_norm)
            
            N_valid = np.count_nonzero(targets_norm)
            targets_norm[targets_norm==0]=math.inf
            
            # Input Relative norm
            # inp_discrepancies_norm_rel = discrepancies_inp_norm / targets_norm
            inp_discrepancies_norm_rel = inputs_PRE_norm / targets_norm
            inp_discrepancies_norm_rel_ave[:,i] = np.sum(inp_discrepancies_norm_rel)/N_valid
            print('input average PRE:' + str(inp_discrepancies_norm_rel_ave))
            inp_discrepancies_norm_rel[inp_discrepancies_norm_rel>=0.3]=0
            norms = axs[2, 2].imshow(inp_discrepancies_norm_rel, cmap='viridis')
            # axs[2, 2].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[2, 2].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[2, 2].get_xaxis().set_visible(False)
            axs[2, 2].get_yaxis().set_visible(False)
            axs[2, 2].set_title('Input Relative norm', fontsize=60, color='grey')
            
            # Prediction Relative norm
            # discrepancies_norm_rel = discrepancies_pred_norm / targets_norm
            discrepancies_norm_rel = predictions_PRE_norm/targets_norm
            discrepancies_norm_rel_ave[:,i] = np.sum(discrepancies_norm_rel)/N_valid
            print(N_valid)
            print('prediction average PRE:' + str(discrepancies_norm_rel_ave))
            discrepancies_norm_rel[discrepancies_norm_rel>=0.3]=0
            norms = axs[2, 3].imshow(discrepancies_norm_rel, cmap='viridis')
            cbar_dis = fig.colorbar(norms, ax=axs[2, 2:4], fraction=0.046, pad=0.04)
            cbar_dis.ax.tick_params(labelsize=50, colors='grey')
            # axs[2, 3].xaxis.set_tick_params(labelsize=30, colors='grey')
            # axs[2, 3].yaxis.set_tick_params(labelsize=30, colors='grey')
            axs[2, 3].get_xaxis().set_visible(False)
            axs[2, 3].get_yaxis().set_visible(False)
            axs[2, 3].set_title('Predicted Relative norm', fontsize=60, color='grey')
            
        # Hide the axes of unused subplots
        axs[2, 0].axis('off')
        # axs[2, 2].axis('off')
        # axs[0, 2].axis('off')
        # axs[2, 0].axis('off')
        # axs[2, 1].axis('off')
        
        # plt.tight_layout()
        plt.savefig('{}/{}_{}.svg'.format(save_dir, model.name, i))
        plt.close()
    print(np.sum(discrepancies_norm_rel_ave)/num)
    print(np.sum(inp_discrepancies_norm_rel_ave)/num)
    