"""
VAE for predicting the velocity fields of 2D fluidic pipes

Reference:
    Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. 
    arXiv preprint arXiv:1312.6114.

Author(s): Wei Chen (wchen459@umd.edu)
           Jun Wang (jwang38@umd.edu)
"""

import time
import numpy as np
import tensorflow as tf

from visualization import visualize


def preprocess(X):
    return X.astype(np.float32)

def postprocess(X):
    X = np.squeeze(X)
    return X

EPSILON = 1e-7

class Model(object):
    
    def __init__(self, resolution=512, input_channel=3, output_channel=2):

        self.name = 'ae_new'
        self.rez = resolution
        self.input_channel = input_channel
        self.output_channel = output_channel
        
    def encoder(self, x, is_training, reuse=tf.AUTO_REUSE):
        
        depth = 32
        kernel_size = (4,4)
        strides = (2,2)
        
        with tf.variable_scope('Encoder', reuse=reuse):
        
            # rez/2 x rez/2 x depth*1
            x = tf.layers.conv2d(x, depth*1, kernel_size, strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
            
            # rez/4 x rez/4 x depth*2
            x = tf.layers.conv2d(x, depth*2, kernel_size, strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
            
            # rez/8 x rez/8 x depth*4
            x = tf.layers.conv2d(x, depth*4, kernel_size, strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
            
            # rez/16 x rez/16 x depth*8
            x = tf.layers.conv2d(x, depth*8, kernel_size, strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
            
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 1024)
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
            
            x = tf.layers.dense(x, 128)
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
            
            return x
        
    def decoder(self, x, is_training, reuse=tf.AUTO_REUSE):
        
        depth = 32
#        dim = int(self.rez/16)
        dim = 4
        kernel_size = (4,4)
        strides = (2,2)
        
        with tf.variable_scope('Decoder', reuse=reuse):
            
            x = tf.layers.dense(x, 1024)
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
    
            x = tf.layers.dense(x, dim*dim*depth*8)
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
            # rez/16 x rez/16 x depth*8
            x = tf.reshape(x, (-1, dim, dim, depth*8))
            
            # rez/8 x rez/8 x depth*4
            x = tf.layers.conv2d_transpose(x, depth*4, kernel_size, strides, padding='same') 
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
            
            # rez/4 x rez/4 x depth*2
            x = tf.layers.conv2d_transpose(x, depth*2, kernel_size, strides, padding='same') 
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
            
            # rez/2 x rez/2 x depth*1
            x = tf.layers.conv2d_transpose(x, depth*1, kernel_size, strides, padding='same') 
            x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.tanh(x)
            
            # rez x rez x output_channel
            x = tf.layers.conv2d_transpose(x, self.output_channel, kernel_size, strides, padding='same')
            x = tf.identity(x, name='outputs')
            
            return x
        
    def train(self, X_train, Y_train, X_test, Y_test, train_steps=2000, batch_size=32, save_interval=0, save_dir=None):
            
        X_train = preprocess(X_train)
        Y_train = preprocess(Y_train)
        X_test = preprocess(X_test)
        Y_test = preprocess(Y_test)
        
        # Train/evaluate switch
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        
        # Inputs
        self.x_in = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.input_channel], name='inputs')
#        x_in_test = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.input_channel], name='inputs_test')
#        self.x_in_resized = tf.image.resize_images(self.x_in, (64, 64))
#        self.x_in_resized_test = tf.image.resize_images(x_in_test, (64, 64))
        
        # Outputs
#        self.x_out = self.decoder(self.encoder(self.x_in_resized, True), True)
#        x_out_test = self.decoder(self.encoder(self.x_in_resized_test, True), True)
        self.x_out = self.decoder(self.encoder(self.x_in, self.is_train), self.is_train)
#        x_out_test = self.decoder(self.encoder(x_in_test, False), False)
        
        # Labels
        self.x_label = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.output_channel], name='labels')
#        x_label_test = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez, self.output_channel], name='labels_test')
#        self.x_label_resized = tf.image.resize_images(self.x_label, (64, 64))
#        x_label_resized_test = tf.image.resize_images(x_label_test, (64, 64))
        
        # Loss
#        loss = tf.losses.mean_squared_error(self.x_label_resized, self.x_out)
##        _, loss_test = tf.metrics.mean_relative_error(x_label_resized_test, x_out_test, x_label_resized_test)
#        _, loss_test = tf.metrics.mean_squared_error(x_label_resized_test, x_out_test)
        
        loss = tf.losses.mean_squared_error(self.x_label, self.x_out)
#        _, loss_test = tf.metrics.mean_relative_error(x_label_resized_test, x_out_test, x_label_resized_test)
        _, loss_test = tf.metrics.mean_squared_error(self.x_label, self.x_out)
        
        # Optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        
        # Encoder variables
        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
        # Decoder variables
        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')
        
        # Training operations
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(loss, var_list=[enc_vars, dec_vars])
        train_op = tf.group([train_op, update_ops])
        
        # Create summaries to monitor losses
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss_test', loss_test)
		
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Start training
        self.sess = tf.Session()
        
        # Run initializers
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('{}/{}/logs'.format(save_dir, self.name), graph=self.sess.graph)
        
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
    
        for t in range(train_steps):
                
            start = time.time()
            
            ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            X_batch = X_train[ind]
            Y_batch = Y_train[ind]
            
            summary_str, _, l = self.sess.run([merged_summary_op, train_op, loss], 
                                              feed_dict={self.x_in: X_batch, self.x_label: Y_batch, self.is_train: True})
    
            l_test = self.sess.run(loss_test, feed_dict={self.x_in: X_test, self.x_label: Y_test, self.is_train: False})
                    
            end = time.time()
            
            summary_writer.add_summary(summary_str, t+1)
            
            # Show messages
            log_mesg = "%d: [training loss] %f [test loss] %f [time] %.2fs" % (t+1, l, l_test, end-start)
            print(log_mesg)
            
            if save_interval>0 and (t+1)%save_interval==0:
                
                # Save the variables to disk.
                save_path = saver.save(self.sess, '{}/{}/model'.format(save_dir, self.name))
                print('Model saved in path: %s' % save_path)
                
                # Plot results
                visualize(5, X_test, Y_test, self, save_dir)
                
        summary_writer.close()
        # Stop the threads
        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)
            
    def restore(self, load_dir=None):
        self.sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/{}/model.meta'.format(load_dir, self.name))
        saver.restore(self.sess, tf.train.latest_checkpoint('{}/{}'.format(load_dir, self.name)))
        
        # Access and create placeholders variables            
        graph = tf.get_default_graph()
        self.x_in = graph.get_tensor_by_name('inputs:0')
        self.x_out = graph.get_tensor_by_name('Decoder/outputs:0')
        self.x_label = graph.get_tensor_by_name('labels:0')
        self.is_train = graph.get_tensor_by_name('is_train:0')

    def predict(self, inputs):
        inputs = preprocess(inputs)
        outputs = self.sess.run(self.x_out, feed_dict={self.x_in: inputs, self.is_train: False})
        return postprocess(outputs)
