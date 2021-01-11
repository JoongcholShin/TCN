# -*- coding: utf-8 -*-


import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]

class Network:
    """
    A trainable version vgg16.
    """

    def __init__(self, comp_npy_path=None, trainable=True):
        if comp_npy_path is not None:
            self.data_dict = np.load(comp_npy_path, allow_pickle=True, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable


    def build(self, rgb):


        self.conv1_r = self.conv_layer(rgb, 3, 24, 1, "conv1_r")
        self.nm1_r = self.nm(self.conv1_r, 'nm1_r')
        self.relu1_r = self.lrelu(self.nm1_r)
        self.conv2_r = self.conv_layer(self.relu1_r, 24, 24, 1, "conv2_r")
        self.nm2_r = self.nm(self.conv2_r, 'nm2_r')
        self.relu2_r = self.lrelu(self.nm2_r)
        self.conv3_r = self.conv_layer(self.relu2_r, 24, 24, 1, "conv3_r")
        self.nm3_r = self.nm(self.conv3_r, 'nm3_r')
        self.relu3_r = self.lrelu(self.nm3_r)
        self.conv4_r = self.conv_layer(self.relu3_r, 24, 24, 2, "conv4_r")
        self.nm4_r = self.nm(self.conv4_r, 'nm4_r')
        self.relu4_r = self.lrelu(self.nm4_r)
        self.conv5_r = self.conv_layer(self.relu4_r, 24, 24, 4, "conv5_r")
        self.nm5_r = self.nm(self.conv5_r, 'nm5_r')
        self.relu5_r = self.lrelu(self.nm5_r)
        self.conv6_r = self.conv_layer(self.relu5_r, 24, 24, 8, "conv6_r")
        self.nm6_r = self.nm(self.conv6_r, 'nm6_r')
        self.relu6_r = self.lrelu(self.nm6_r)
        self.conv7_r = self.conv_layer(self.relu6_r, 24, 24, 16, "conv7_r")
        self.nm7_r = self.nm(self.conv7_r, 'nm7_r')
        self.relu7_r = self.lrelu(self.nm7_r)
        self.conv9_r = self.conv_layer(self.relu7_r, 24, 24, 1, "conv9_r")
        self.nm9_r = self.nm(self.conv9_r, 'nm9_r')
        self.relu9_r = self.lrelu(self.nm9_r)
        self.conv10_r = self.conv_layer_last(self.relu9_r, 24, 3, "conv10_r")
        self.R_out = self.conv10_r



        self.conv1_d = self.conv_layer(rgb, 3, 24, 1, "conv1_d")
        self.nm1_d = self.nm(self.conv1_d, 'nm1_d')
        self.relu1_d = self.lrelu(self.nm1_d)
        self.conv2_d = self.conv_layer(self.relu1_d, 24, 24, 1, "conv2_d")
        self.nm2_d = self.nm(self.conv2_d, 'nm2_d')
        self.relu2_d = self.lrelu(self.nm2_d)
        self.conv3_d = self.conv_layer(self.relu2_d, 24, 24, 1, "conv3_d")
        self.nm3_d = self.nm(self.conv3_d, 'nm3_d')
        self.relu3_d = self.lrelu(self.nm3_d)
        self.conv4_d = self.conv_layer(self.relu3_d, 24, 24, 2, "conv4_d")
        self.nm4_d = self.nm(self.conv4_d, 'nm4_d')
        self.relu4_d = self.lrelu(self.nm4_d)
        self.conv5_d = self.conv_layer(self.relu4_d, 24, 24, 4, "conv5_d")
        self.nm5_d = self.nm(self.conv5_d, 'nm5_d')
        self.relu5_d = self.lrelu(self.nm5_d)
        self.conv6_d = self.conv_layer(self.relu5_d, 24, 24, 8, "conv6_d")
        self.nm6_d = self.nm(self.conv6_d, 'nm6_d')
        self.relu6_d = self.lrelu(self.nm6_d)
        self.conv7_d = self.conv_layer(self.relu6_d, 24, 24, 16, "conv7_d")
        self.nm7_d = self.nm(self.conv7_d, 'nm7_d')
        self.relu7_d = self.lrelu(self.nm7_d)
        self.conv9_d = self.conv_layer(self.relu7_d, 24, 24, 1, "conv9_d")
        self.nm9_d = self.nm(self.conv9_d, 'nm9_d')
        self.relu9_d = self.lrelu(self.nm9_d)
        self.conv10_d = self.conv_layer_last(self.relu9_d, 24, 3, "conv10_d")
        self.D_out = self.conv10_d


        Fin = tf.concat([self.R_out, self.D_out], 3)


        self.conv1_GS = self.conv_layer(Fin, 6, 24, 1, "conv1_GS")
        self.nm1_GS = self.bn(self.conv1_GS, 'nm1_GS')
        self.relu1_GS = self.lrelu(self.nm1_GS)



        self.conv2_GS = self.conv_layer(self.relu1_GS, 24, 24, 1, "conv2_GS")
        self.nm2_GS = self.bn(self.conv2_GS, 'nm2_GS')
        self.relu2_GS = self.lrelu(self.nm2_GS)


        self.conv3_GS = self.conv_layer(self.relu2_GS, 24, 24, 1, "conv3_GS")
        self.nm3_GS = self.bn(self.conv3_GS, 'nm3_GS')
        self.relu3_GS = self.lrelu(self.nm3_GS)


        self.conv4_GS = self.conv_layer(self.relu3_GS, 24, 24, 2, "conv4_GS")

        self.nm4_GS = self.bn(self.conv4_GS, 'nm4_GS')
        self.relu4_GS = self.lrelu(self.nm4_GS)


        self.conv5_GS = self.conv_layer(self.relu4_GS, 24, 24, 4, "conv5_GS")
        self.nm5_GS = self.bn(self.conv5_GS, 'nm5_GS')
        self.relu5_GS = self.lrelu(self.nm5_GS)


        self.conv6_GS = self.conv_layer(self.relu5_GS, 24, 24, 8, "conv6_GS")
        self.nm6_GS = self.bn(self.conv6_GS, 'nm6_GS')
        self.relu6_GS = self.lrelu(self.nm6_GS)


        self.conv7_GS = self.conv_layer(self.relu6_GS, 24, 24, 16, "conv7_GS")
        self.nm7_GS = self.bn(self.conv7_GS, 'nm7_GS')
        self.relu7_GS = self.lrelu(self.nm7_GS)


        self.conv9_GS = self.conv_layer(self.relu7_GS, 24, 24, 1, "conv9_GS")
        self.nm9_GS = self.bn(self.conv9_GS, 'nm9_GS')
        self.relu9_GS = self.lrelu(self.nm9_GS)


        self.conv10_GS = self.conv_layer_last(self.relu9_GS, 24, 1, "conv10_GS")
        self.att = tf.nn.sigmoid(self.conv10_GS)



        self.residual=(self.conv10_d*self.att + self.conv10_r*(1-self.att))
        self.F_out=self.residual








    def nm(self,x,name):
        initial_value = tf.constant(1.0, dtype=tf.float32)
        w0 = self.get_var(initial_value, name, 0, name + "aw0")

        initial_value = tf.constant(0.0, dtype=tf.float32)
        w1 = self.get_var(initial_value, name, 1, name + "aw1")

        initial_value = tf.zeros([x.get_shape()[-1]], dtype=tf.float32)
        b0 = self.get_var(initial_value, name, 2, name + "bw0")
        initial_value = tf.ones([x.get_shape()[-1]], dtype=tf.float32)
        b1 = self.get_var(initial_value, name, 3, name + "bw1")
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        x_hat=tf.nn.batch_normalization(x,batch_mean,batch_var,b0,b1,0.000001)


        return w1 * x + w0 * x_hat


    def bn(self,x, name):
        initial_value = tf.zeros([x.get_shape()[-1]], dtype=tf.float32)
        w0 = self.get_var(initial_value, name, 0, name + "bw0")
        initial_value = tf.ones([x.get_shape()[-1]], dtype=tf.float32)
        w1 = self.get_var(initial_value, name, 1, name + "bw1")
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        x_hat = tf.nn.batch_normalization(x, batch_mean, batch_var, w0, w1, 0.000001)

        return x_hat

    def bn_FC(self,x, name):
        initial_value = tf.zeros([x.get_shape()[-1]], dtype=tf.float32)
        w0 = self.get_var(initial_value, name, 0, name + "bw0")
        initial_value = tf.ones([x.get_shape()[-1]], dtype=tf.float32)
        w1 = self.get_var(initial_value, name, 1, name + "bw1")
        batch_mean, batch_var = tf.nn.moments(x, [0])
        x_hat = tf.nn.batch_normalization(x, batch_mean, batch_var, w0, w1, 0.000001)

        return x_hat


    def lrelu(self,x):
        return tf.maximum(x * 0.2, x)




    def conv_layer(self, bottom, in_channels, out_channels, r, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv=tf.nn.atrous_conv2d(bottom, filt, r, padding="SAME")
            bias = tf.nn.bias_add(conv, conv_biases)

            return bias

    def conv_layer_last(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(1, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom,filt,[1,1,1,1],padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)


            return bias




    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.0001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")
        initial_value = tf.truncated_normal([out_channels], .0, .0001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")
        return filters, biases



    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value
            print("initial")

        if self.trainable:
            var = tf.Variable(value, name=var_name, dtype=tf.float32)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

#        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./CAN-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out
        np.save(npy_path, data_dict)
        print("saved")

        return npy_path


    def get_var_count(self):  # variable 개수  counting
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

