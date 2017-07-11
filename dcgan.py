
from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from ops import *

def deconv_size(size, stride):
    # padding outside by 1, with dilation size 1 , 3x3 -> 7x7 then filter with 3x3 -> 5x5
    # apply (W - F + 2P) / S = output_size
    # 64 -> 32 with stride 2
    return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
    def __init__(self, sess, batch_size=100, input_height=28, input_width=28, output_height=28, output_width=28,
        z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=1,
        dataset_name='mnist', checkpoint_dir=None, sample_dir=None):

        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """

        self.sess = sess

        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        # variables for building G & D
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim


        self.g_batch_norm_0 = batch_norm(name = "g_batch_norm_0")
        self.g_batch_norm_1 = batch_norm(name = "g_batch_norm_1")
        self.g_batch_norm_2 = batch_norm(name = "g_batch_norm_2")
        self.g_batch_norm_3 = batch_norm(name = "g_batch_norm_3")

        self.d_batch_norm_0 = batch_norm(name = "d_batch_norm_0")
        self.d_batch_norm_1 = batch_norm(name = "d_batch_norm_1")
        self.d_batch_norm_2 = batch_norm(name = "d_batch_norm_2")
        self.d_batch_norm_3 = batch_norm(name = "d_batch_norm_3")

        self.final_channel = c_dim
        self.dataset_name = dataset_name

        if self.dataset_name == 'mnist':
            self.input_sample = input_data.read_data_sets("./mnist/data/", one_hot=True)
            self.total_sample = self.input_sample.train.num_examples
        
        self.n_input = self.input_height * self.input_width * self.final_channel


        self.build_model()


    def get_noise(self, batch_size, n_noise):
        return np.random.uniform(-0.25, +0.25, size=(batch_size, n_noise))

    def build_model(self):
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        self.x = tf.placeholder(tf.float32, [None, self.n_input])

        self.G_z = self.generator(self.z)
        self.D_G_z = self.discriminator(self.G_z)
        self.D_x = self.discriminator(self.x, reuse = True)

        self.loss_D = tf.reduce_mean(tf.log(self.D_x) + tf.log(1 - self.D_G_z))
        self.loss_G = tf.reduce_mean(tf.log(self.D_G_z))

        t_var = tf.trainable_variables()

        self.g_vars = [v for v in t_var if 'g_' in v.name]
        self.d_vars = [v for v in t_var if 'd_' in v.name]
        print(self.g_vars)
        print(self.d_vars)

        self.saver = tf.train.Saver(max_to_keep = 1)

    def train(self, config, total_epoch=100, sample_size = 10):
        self.train_D = tf.train.AdamOptimizer(1e-4, beta1=0.5)\
                                .minimize(-self.loss_D, var_list = self.d_vars)
        self.train_G = tf.train.AdamOptimizer(2e-4, beta1=0.5)\
                                .minimize(-self.loss_G, var_list = self.g_vars)

        total_batch = int(self.total_sample/self.batch_size)
        loss_val_D, loss_val_G = 0, 0


        self.sess.run(tf.global_variables_initializer())
        for epoch in range(total_epoch):
            for i in range(total_batch):
                batch_x, batch_y = self.input_sample.train.next_batch(self.batch_size)
                noise = self.get_noise(self.batch_size, self.z_dim)

                _, loss_val_D = self.sess.run([self.train_D, self.loss_D], feed_dict = {self.x: batch_x, self.z: noise})
                _, loss_val_G = self.sess.run([self.train_G, self.loss_G], feed_dict = {self.z: noise})

            print('Epoch: ', '%04d' % epoch,
                  'D loss: {:.4}'.format(loss_val_D),
                  'G loss: {:.4}'.format(loss_val_G))

            #if epoch == 0 or (epoch + 1) % 10 == 0:
            noise = self.get_noise(self.batch_size, self.z_dim)
            samples = self.sess.run(self.G_z, feed_dict={self.z: noise})

            fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

            for i in range(sample_size):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(samples[i], (28, 28)), cmap='Greys')

            plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)





    def generator(self, z):
        # final result should be size of self.output_height, self.output_width
        # we suppose 4 conv layer and each strides with 2
        g_h4, g_w4 = deconv_size(self.output_height, 2), deconv_size(self.output_width, 2)
        g_h3, g_w3 = deconv_size(g_h4, 2), deconv_size(g_w4, 2)
        g_h2, g_w2 = deconv_size(g_h3, 2), deconv_size(g_w3, 2)
        g_h1, g_w1 = deconv_size(g_h2, 2), deconv_size(g_w2, 2)


        with tf.variable_scope('generator') as scope:
            self.noise = tf.nn.relu(linear(z, g_h1 * g_w1 * self.gfc_dim, 'g_lin_1'))
            self.noise_reshape = tf.reshape(self.noise, [-1, g_h1, g_w1, self.gfc_dim])
            self.noise_batch_relu = tf.nn.relu(self.g_batch_norm_0(self.noise_reshape))

            self.deconv_1 = deconv2d(self.noise_batch_relu, [self.batch_size, g_h2, g_w2, int(self.gfc_dim / 2)], name = 'g_deconv_1')
            self.deconv_1_batch_relu = tf.nn.relu(self.g_batch_norm_1(self.deconv_1))

            self.deconv_2 = deconv2d(self.deconv_1_batch_relu, [self.batch_size, g_h3, g_w3, int(self.gfc_dim / 4)], name = 'g_deconv_2')
            self.deconv_2_batch_relu = tf.nn.relu(self.g_batch_norm_2(self.deconv_2))

            self.deconv_3 = deconv2d(self.deconv_2_batch_relu, [self.batch_size, g_h4, g_w4, int(self.gfc_dim / 8)], name = 'g_deconv_3')
            self.deconv_3_batch_relu = tf.nn.relu(self.g_batch_norm_3(self.deconv_3))

            self.deconv_4 = deconv2d(self.deconv_3_batch_relu, [self.batch_size, self.output_height, self.output_width, self.final_channel], name = 'g_deconv_4')
            self.deconv_4_tanh = tf.nn.tanh(self.deconv_4)
            print('shape of generated tensor : {}'.format(self.deconv_4_tanh)) 

            return self.deconv_4_tanh

    # reuse true for sharing variables between discriminating generated image and sample image
    # discriminator is called twice so reuse it for second call
    def discriminator(self, image, reuse = False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            # reshape the image if needed
            image = tf.reshape(image, [-1, self.input_height, self.input_height, self.final_channel])

            self.conv_1 = conv2d(image, self.df_dim, name = 'd_conv_1')
            self.conv_1_batch_lrelu = leakyReLU(self.d_batch_norm_0(self.conv_1))

            self.conv_2 = conv2d(self.conv_1_batch_lrelu, self.df_dim * 2, name = 'd_conv_2')
            self.conv_2_batch_lrelu = leakyReLU(self.d_batch_norm_1(self.conv_2))

            self.conv_3 = conv2d(self.conv_2_batch_lrelu, self.df_dim * 4, name = 'd_conv_3')
            self.conv_3_batch_lrelu = leakyReLU(self.d_batch_norm_2(self.conv_3))

            self.conv_4 = conv2d(self.conv_3_batch_lrelu, self.df_dim * 8, name = 'd_conv_4')
            self.conv_4_batch_lrelu = leakyReLU(self.d_batch_norm_3(self.conv_4))

            d_linear = linear(tf.reshape(self.conv_4_batch_lrelu, [-1, 8192]), 1, 'd_linear')

            return tf.nn.sigmoid(d_linear)
