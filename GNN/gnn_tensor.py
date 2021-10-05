"""
GNN for composing pairs of 2D fluidic pipes

Reference:
    Deepmind (2018). Relational inductive biases, deep learning, and graph networks.
    arXiv:1806.01261

Author: Jun Wang (jwang38@umd.edu)
Created on Fri Jan 24 14:15:08 2020
"""

# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models

import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import time

# Fuzz factor
EPSILON = 1e-7

class GNN_Model(object):
    
    def __init__(self, resolution=64, node_input_channel=2, edge_size= 1, nodes_output_size=64*64*2, num_processing_steps_tr=1, num_processing_steps_te=1):
        
        self.name = 'gnn_tensor'
        self.rez = resolution
        self.node_input_channel = node_input_channel
        self.edge_size = edge_size
        self.nodes_output_size = nodes_output_size
        self.num_processing_steps_tr = num_processing_steps_tr
        self.num_processing_steps_te = num_processing_steps_te
        
    def two_node_graph(self, i, nodes_first, nodes_next, edges_all):
    # def two_node_graph(i, nodes_first, nodes_next):
        """Define a simple connection between two pipes (A graph of two nodes connected by an edge)
    
        These are two velocity fields (64 X 64 X 2) connected by an edge.
    
        Args:
        i: i-th two-node graph
        nodes_first: features of the first node for the first pipe in the two-node graph
        nodes_next: features of the second node for the last pipe in the two-node graph
        edges_all: features of the edge
    
        Returns:
        data_dict: dictionary with globals, nodes, edges, receivers and senders
            to represent a two-node graph.
            
        Nodes: Velocity fields (64 X 64 X 2) of two connected pipes.
        Edges: If two velocity fields are connected horizontally, the edge feature is 0.
               If two velocity fields are connected vertically, the edge feature is 1.
        Globals: Global feature is set to 0.
        """
        
        nodes_first = tf.reshape(nodes_first[i, :, :, :], [-1])
        nodes_next = tf.reshape(nodes_next[i, :, :, :], [-1])

        nodes = tf.stack([nodes_first, nodes_next], axis=0)

#         edges = edges_all[i, :]
        edges = tf.reshape(edges_all[i, :], [1, 1])

        senders = [0]
        receivers = [1]

        return {
            "globals": [0.],
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers
        }

    def create_loss_ops_tr(self, target_ops_tr, output_ops_tr):
        """
        Create loss operations from targets and outputs in training.
        
        Args:
        target_ops_tr: The target graphs in training data.
        output_ops_tr: The list of output graphs from the model.

        Returns:
        A list of loss values (tf.Tensor), one per output op.
        """
    #     loss_ops = [
    #       tf.reduce_mean(
    #           tf.reduce_sum((output_op.nodes - target_ops.nodes)**2, axis=-1))
    #       for output_op in output_ops
    #     ]

        for output_op_tr in output_ops_tr:
            loss_ops_tr = tf.losses.mean_squared_error(output_op_tr.nodes, target_ops_tr.nodes)

        return loss_ops_tr

    def create_loss_ops_te(self, target_ops_te, output_ops_te):
        """
        Create loss operations from targets and outputs in evaluation.
        
        Args:
        target_ops_te: The target graphs in testing data.
        output_ops_te: The list of output graphs from the model.

        Returns:
        A list of loss values (tf.Tensor), one per output op.
        """
        
        for output_op_te in output_ops_te:
            _, loss_ops_te = tf.metrics.mean_squared_error(output_op_te.nodes, target_ops_te.nodes)

        return loss_ops_te

    def train(self, X_train_first, X_train_next, Y_train_first, Y_train_next, X_test_first, X_test_next, Y_test_first, Y_test_next, edge_train, edge_test,
              train_steps, batch_size, save_interval, save_dir=None):
        '''
        Define the training operation for GNN model training.

        Args:
        X_train_first : Input first nodes for training.
        X_train_next: Input second nodes for training.
        Y_train_first: Target first nodes for training.
        Y_train_next: Target second nodes for training.
        X_test_first: Input first nodes for testing.
        X_test_next: Input second nodes for testing.
        Y_test_first: Target first nodes for testing.
        Y_test_next: Target second nodes for testing.
        edge_train: Edge features for training.
        edge_test: Edge features for testing.
        '''
        
        # Define the placeholders for GNN nodes and edges
        # Placeholders for training
        self.x_in_train_first = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.node_input_channel], name='inputs_first_tr')
        self.x_in_train_next = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.node_input_channel], name='inputs_next_tr')

        self.edge_train = tf.placeholder(tf.float32, shape=[None, self.edge_size], name='inputs_edge_tr')

        self.x_tar_train_first = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.node_input_channel], name='targets_first_tr')
        self.x_tar_train_next = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.node_input_channel], name='targets_next_tr')

        # Placeholders for testing
        self.x_in_test_first = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.node_input_channel], name='inputs_first_te')
        self.x_in_test_next = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.node_input_channel], name='inputs_next_te')

        self.edge_test = tf.placeholder(tf.float32, shape=[None, self.edge_size], name='inputs_edge_te')

        self.x_tar_test_first = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.node_input_channel], name='targets_first_te')
        self.x_tar_test_next = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.node_input_channel], name='targets_next_te')
        
        # Create the input GNN graphs for training (training batch size is 32)
        two_node_input_graph_tr_dict = [self.two_node_graph(i, self.x_in_train_first, self.x_in_train_next, self.edge_train) for i in range(batch_size)]
        two_node_input_graph_tr = utils_tf.data_dicts_to_graphs_tuple(two_node_input_graph_tr_dict)
        
        # Create the target GNN graphs for training (training batch size is 32)
        two_node_target_graph_tr_dict = [self.two_node_graph(i, self.x_tar_train_first, self.x_tar_train_next, self.edge_train) for i in range(batch_size)]
        two_node_target_graph_tr = utils_tf.data_dicts_to_graphs_tuple(two_node_target_graph_tr_dict)

        # Create the input GNN graphs for testing
        two_node_input_graph_te_dict = [self.two_node_graph(i, self.x_in_test_first, self.x_in_test_next, self.edge_test) for i in range(X_test_first.shape[0])]
        two_node_input_graph_te = utils_tf.data_dicts_to_graphs_tuple(two_node_input_graph_te_dict)

        # Create the target GNN graphs for testing
        two_node_target_graph_te_dict = [self.two_node_graph(i, self.x_tar_test_first, self.x_tar_test_next, self.edge_test) for i in range(X_test_first.shape[0])]
        two_node_target_graph_te = utils_tf.data_dicts_to_graphs_tuple(two_node_target_graph_te_dict)  
        
        # The encode-process-decode GNN model in graphnets DeepMind library
        model = models.EncodeProcessDecode(node_output_size = self.nodes_output_size)
        
        # Generate the output graphs for training and testing using the GNN model above
        two_node_output_graph_tr = model(two_node_input_graph_tr, self.num_processing_steps_tr)
        two_node_output_graph_te = model(two_node_input_graph_te, self.num_processing_steps_te)
#         self.two_node_output_graph = tf.tuple(two_node_output_graph_te[0].nodes, name='outputs')
        self.two_node_output_graph = tf.identity(two_node_output_graph_te[0].nodes, name='outputs')

        # Calculate the losses in the training and evaluation processes
        loss_ops_tr = self.create_loss_ops_tr(two_node_target_graph_tr, two_node_output_graph_tr)
        loss_ops_te = self.create_loss_ops_te(two_node_target_graph_te, two_node_output_graph_te)
        
        # Minimize the training loss
        learning_rate = 1e-3
        optimizer = tf.train.AdamOptimizer(learning_rate)
        step_op = optimizer.minimize(loss_ops_tr)
        
        # Create summaries to monitor losses
        tf.summary.scalar('loss_tr', loss_ops_tr)
        tf.summary.scalar('loss_te', loss_ops_te)
        
        # Merge all summaries into a single op
#         merged = tf.summary.merge([loss_tr_summary, loss_te_summary])
        merged = tf.summary.merge_all()

        # Add ops to save and restore all the variables
        saver = tf.train.Saver()

#         try:
#             sess.close()
#         except NameError:
#             pass

        # Create training session 
        self.sess = tf.Session()

        # Run initializers
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    #     last_iteration = 0
    #     logged_iterations = []
    #     losses_tr = []
    #     losses_te = []

        # save_dir = '/home/junwang/Documents/DeepCFD/Spyder/GNN/model_StrL'

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('{}/{}/logs'.format(save_dir, self.name), graph=self.sess.graph)
        
        # Start training
        for iteration in range(train_steps):
            
            start = time.time()

            ind = np.random.choice(X_train_first.shape[0], size=batch_size, replace=False)
            
            # Load the data batch
            all_input_first_train_batch = X_train_first[ind]
            all_input_next_train_batch = X_train_next[ind]

            all_output_first_train_batch = Y_train_first[ind]
            all_output_next_train_batch = Y_train_next[ind]

            all_edge_feature_train_batch = edge_train[ind]

#             summary_str, _, loss_train, loss_test, outputs, predicted_test = self.sess.run([merged, step_op, loss_ops_tr, loss_ops_te, two_node_output_graph_tr, two_node_output_graph_te],
#                                                                            feed_dict={self.x_in_train_first: all_input_first_train_batch, self.x_in_train_next: all_input_next_train_batch,
#                                                                                       self.x_tar_train_first: all_output_first_train_batch, self.x_tar_train_next: all_output_next_train_batch,
#                                                                                       self.x_in_test_first: X_test_first, self.x_in_test_next: X_test_next,
#                                                                                       self.x_tar_test_first: Y_test_first, self.x_tar_test_next: Y_test_next,
#                                                                                       self.edge_train: all_edge_feature_train_batch, self.edge_test: edge_test})
            
            # Run all the needed training ops
            summary_str, _, loss_train, loss_test = self.sess.run([merged, step_op, loss_ops_tr, loss_ops_te],
                                                                           feed_dict={self.x_in_train_first: all_input_first_train_batch, self.x_in_train_next: all_input_next_train_batch,
                                                                                      self.x_tar_train_first: all_output_first_train_batch, self.x_tar_train_next: all_output_next_train_batch,
                                                                                      self.x_in_test_first: X_test_first, self.x_in_test_next: X_test_next,
                                                                                      self.x_tar_test_first: Y_test_first, self.x_tar_test_next: Y_test_next,
                                                                                      self.edge_train: all_edge_feature_train_batch, self.edge_test: edge_test})

#             loss_test, predicted_test = self.sess.run([loss_ops_te, two_node_output_graph_te],
#                                                       feed_dict={self.x_in_test_first: X_test_first, self.x_in_test_next: X_test_next,
#                                                                  self.x_tar_test_first: Y_test_first, self.x_tar_test_next: Y_test_next,
#                                                                  self.edge_test: edge_test})

            end = time.time()

            summary_writer.add_summary(summary_str, iteration + 1)
            
            # Show messages
            log_mesg = "%d: [training loss] %f [test loss] %f [time] %.2fs" % (iteration+1, loss_train, loss_test, end-start)
            print(log_mesg)

            if iteration%save_interval==0:
                # Save the variables to disk
        #         save_path = saver.save(sess, '{}/model'.format(save_dir))
                save_path = saver.save(self.sess, '{}/{}/gnnmodel'.format(save_dir, self.name))
                print('Model saved in path: %s' % save_path)
            
        summary_writer.close()
        
    def restore(self, load_dir=None):
        # Create session for restore
        self.sess = tf.Session()
        
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/{}/gnnmodel.meta'.format(load_dir, self.name))
        saver.restore(self.sess, tf.train.latest_checkpoint('{}/{}'.format(load_dir, self.name)))
        
        # Access and create placeholders variables
        graph = tf.get_default_graph()
        self.x_in_train_first = graph.get_tensor_by_name('inputs_first_tr:0')
        self.x_in_train_next = graph.get_tensor_by_name('inputs_next_tr:0')
        self.x_tar_train_first = graph.get_tensor_by_name('targets_first_tr:0')
        self.x_tar_train_next = graph.get_tensor_by_name('targets_next_tr:0')
        self.x_in_test_first = graph.get_tensor_by_name('inputs_first_te:0')
        self.x_in_test_next = graph.get_tensor_by_name('inputs_next_te:0')
        self.x_tar_test_first = graph.get_tensor_by_name('targets_first_te:0')
        self.x_tar_test_next = graph.get_tensor_by_name('targets_next_te:0')
        self.edge_train = graph.get_tensor_by_name('inputs_edge_tr:0')
        self.edge_test = graph.get_tensor_by_name('inputs_edge_te:0')
        self.two_node_output_graph = graph.get_tensor_by_name('outputs:0')
        
    def predict(self, inputs_first, inputs_next, inputs_edge):
        outputs = self.sess.run(self.two_node_output_graph, feed_dict={self.x_in_test_first: inputs_first, self.x_in_test_next: inputs_next, self.edge_test: inputs_edge})
        return outputs
        

