"""
Trains a model, and visulizes results

Author(s): Wei Chen (wchen459@umd.edu)
           Jun Wang (jwang38@umd.edu)
"""

import argparse
import os
import glob
import numpy as np
from importlib import import_module
import shutil
import h5py

from visualization import visualize_samples, visualize


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train/Evaluate')
    parser.add_argument('mode', type=str, default='startover', help='startover or evaluate')
    parser.add_argument('model', type=str, default='ae_new', help='model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--save_interval', type=int, default=500, help='save interval')
    parser.add_argument('--train_steps', type=int, default=200000, help='training steps')
    args = parser.parse_args()
    assert args.mode in ['startover', 'evaluate']
    
    if args.mode == 'startover':
        training_dir = './saved_model_StrL_dis'
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
        log_dir = '{}/{}/logs'.format(training_dir, args.model)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        
    results_dir = './results_StrL_dis'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    hf_train = h5py.File('./DeCNN_database_StrL_dis/train.h5', 'r')
    hf_test = h5py.File('./DeCNN_database_StrL_dis/test.h5', 'r')
    
    X_train = np.array(hf_train.get('train_x'))
    Y_train = np.array(hf_train.get('train_y'))
    
    X_test = np.array(hf_test.get('test_x'))
    Y_test = np.array(hf_test.get('test_y'))
      
    rez = X_train.shape[1]
#    print(rez)
    input_channel = X_train.shape[3]
#    print(input_channel)
    output_channel = Y_train.shape[3]
#    print(output_channel)
    
    # Train/Evaluate
    m = import_module(args.model)
    model = m.Model(rez, input_channel, output_channel)
    model_dir = './saved_model_StrL_dis'
    if args.mode == 'startover':
        print('Start training ...')
        model.train(X_train, Y_train, X_test, Y_test, batch_size=args.batch_size, train_steps=args.train_steps, 
                    save_interval=args.save_interval, save_dir=model_dir)
    else:
        print('Evaluating ...')
        model.restore(model_dir)
        
    # Visulize Results
    print('Plotting results ...')
    visualize(20, X_test, Y_test, model, results_dir)

    print('All completed :)')
