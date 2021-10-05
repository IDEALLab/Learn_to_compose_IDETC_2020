"""
GNN for connection between two consecutive pipe flows
Created on Fri Jan 24 15:20:16 2020

Author: Jun Wang (jwang38@umd.edu)
"""

# Imports
import argparse
import os
#import glob
import numpy as np
from importlib import import_module
import shutil
import h5py
import tensorflow as tf

from visualization_tensor import visualize

if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train/Evaluate')
    parser.add_argument('mode', type=str, default='train', help='train or evaluate')
    parser.add_argument('model', type=str, default='gnn_tensor', help='model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--save_interval', type=int, default=500, help='save interval')
    parser.add_argument('--train_steps', type=int, default=1000000, help='training steps')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    if args.mode == 'train':
        training_dir = './saved_model_tensor_StrL_new'
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
        log_dir = '{}/{}/logs'.format(training_dir, args.model)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        
    # results_dir = './results_tensor_StrL'
    results_dir = './results_tensor_StrL_new'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Data
    # Training
    # Load node features for training (25600)
    hf_train = h5py.File('./GNN_database_StrL/train_nodes.h5', 'r')
    # Input
    all_input_first_train = np.array(hf_train.get('train_all_input_first'))
    # all_input_first_train = all_input_first_train[0:25600, :, :, :]
    all_input_next_train = np.array(hf_train.get('train_all_input_next'))
    # all_input_next_train = all_input_next_train[0:25600, :, :, :]
    # Target
    all_output_first_train = np.array(hf_train.get('train_all_output_first'))
    # all_output_first_train = all_output_first_train[0:25600, :, :, :]
    all_output_next_train = np.array(hf_train.get('train_all_output_next'))
    # all_output_next_train = all_output_next_train[0:25600, :, :, :]
    
    # Load edge features for training (25600)
    hf_edge_train = h5py.File('./GNN_database_StrL/train_edges.h5', 'r')
    all_edge_feature_train = np.array(hf_edge_train.get('train_edge'))
    # all_edge_feature_train = all_edge_feature_train[0:25600, :]
    
    # two_node_input_graph_tr_dict = [two_node_graph(i, all_input_first_train, all_input_next_train, all_edge_feature_train) for i in range(all_input_first_train.shape[0])]
    # two_node_target_graph_tr_dict = [two_node_graph(i, all_output_first_train, all_output_next_train, all_edge_feature_train) for i in range(all_output_first_train.shape[0])]
    
    # Load node features for testing (6400)
    hf_test = h5py.File('./GNN_database_StrL/test_nodes.h5', 'r')
    # Input
    all_input_first_test = np.array(hf_test.get('test_all_input_first'))
    all_input_first_test = all_input_first_test[0:6400, :, :, :]
    all_input_next_test = np.array(hf_test.get('test_all_input_next'))
    all_input_next_test = all_input_next_test[0:6400, :, :, :]
    # Target
    all_output_first_test = np.array(hf_test.get('test_all_output_first'))
    all_output_first_test = all_output_first_test[0:6400, :, :, :]
    all_output_next_test = np.array(hf_test.get('test_all_output_next'))
    all_output_next_test = all_output_next_test[0:6400, :, :, :]
    
    # Load edge features for testing (6400)
    hf_edge_test = h5py.File('./GNN_database_StrL/test_edges.h5', 'r')
    all_edge_feature_test = np.array(hf_edge_test.get('test_edge'))
    all_edge_feature_test = all_edge_feature_test[0:6400, :]
    
    # two_node_input_graph_te_dict = [two_node_graph(i, all_input_first_test, all_input_next_test, all_edge_feature_test) for i in range(all_input_first_test.shape[0])]
    # two_node_target_graph_te_dict = [two_node_graph(i, all_output_first_test, all_output_next_test, all_edge_feature_test) for i in range(all_output_first_test.shape[0])]
	
	# Testing: Multiple (N) StrTube
    # Load node features for testing multiple StrTube (N-1)
    # Input
    # all_input_first_test_2Str = np.zeros([6400, 64, 64, 2])
    # all_input_first_test_2Str[0:1, :, :, :] = np.load('./GNN_database_StrL/StrTube_3pipes/input_3pipes_StrStr_first.npy')
    # all_input_next_test_2Str = np.zeros([6400, 64, 64, 2])
    # all_input_next_test_2Str[0:1, :, :, :] = np.load('./GNN_database_StrL/StrTube_3pipes/input_3pipes_StrStr_next.npy')
    # # Target
    # all_output_first_test_2Str = np.zeros([6400, 64, 64, 2])
    # all_output_first_test_2Str[0:1, :, :, :] = np.load('./GNN_database_StrL/StrTube_3pipes/3pipes_Field_pipe2.npy')
    # all_output_next_test_2Str = np.zeros([6400, 64, 64, 2])
    # all_output_next_test_2Str[0:1, :, :, :] = np.load('./GNN_database_StrL/StrTube_3pipes/3pipes_Field_pipe3.npy') 
    
    # # Load edge features for testing 9 StrTube (8)
    # all_edge_feature_test_2Str = np.zeros([6400, 1])
    # all_edge_feature_test_2Str[0:1, :] = np.load('./GNN_database_StrL/StrTube_3pipes/edge_3pipes_StrTube.npy')
    
    # Make the loaded data into GNN graphs
    # two_node_input_graph_te_9Str_dict = [two_node_graph(i, all_input_first_test_9Str, all_input_next_test_9Str, all_edge_feature_test_9Str) for i in range(all_input_first_test_9Str.shape[0])]
    # two_node_target_graph_te_9Str_dict = [two_node_graph(i, all_output_first_test_9Str, all_output_next_test_9Str, all_edge_feature_test_9Str) for i in range(all_output_first_test_9Str.shape[0])]
    
    tf.reset_default_graph()
    
    resolution=64
    node_input_channel=2
    edge_size= 1
    nodes_output_size=64*64*2
    num_processing_steps_tr=1
    num_processing_steps_te=1
    
    # Train/Evaluate
    m = import_module(args.model)
    model = m.GNN_Model(resolution, node_input_channel, edge_size, nodes_output_size, num_processing_steps_tr, num_processing_steps_te)
    model_dir = './saved_model_tensor_StrL_new'
    if args.mode == 'train':
        print('Start training ...')
        model.train(X_train_first=all_input_first_train, X_train_next=all_input_next_train, Y_train_first=all_output_first_train, Y_train_next=all_output_next_train, 
                    X_test_first=all_input_first_test, X_test_next=all_input_next_test, Y_test_first=all_output_first_test, Y_test_next=all_output_next_test,
                    edge_train=all_edge_feature_train, edge_test=all_edge_feature_test, train_steps=args.train_steps, batch_size=args.batch_size, save_interval=args.save_interval, save_dir=model_dir)
        
    else:
        print('Evaluating ...')
        model.restore(model_dir)
        
    # Visualize Results
    print('Plotting results ...')
    visualize(20, all_input_first_test, all_output_first_test, all_input_next_test, all_output_next_test, all_edge_feature_test, model, results_dir)
    # visualize(1, all_input_first_test_2Str, all_output_first_test_2Str, all_input_next_test_2Str, all_output_next_test_2Str, all_edge_feature_test_2Str, model, results_dir)
    
    print('All completed :)')
    
