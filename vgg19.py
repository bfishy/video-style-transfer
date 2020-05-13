import numpy as np
import scipy.io
import tensorflow as tf
import utils

# VGG-19 file links
VGG_DOWNLOAD_LINK = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
VGG_FILENAME = "imagenet-vgg-verydeep-19.mat"

# Hand design the VGG-19 network and load the pre-trained weights
class VGG(object):
    def __init__(self, img):
        # Download file
        utils.download(VGG_DOWNLOAD_LINK, VGG_FILENAME)
        # Load and process file
        self.vgg_layers = scipy.io.loadmat(VGG_FILENAME)["layers"]
        self.input = img

        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

    def _weights(self, layer_idx, expected_layer_name):
        W = self.vgg_layers[0][layer_idx][0][0][2][0][0]
        b = self.vgg_layers[0][layer_idx][0][0][2][0][1]
        layer_name = self.vgg_layers[0][layer_idx][0][0][0][0]

        return W, b.reshape(b.size)

    def conv2d_relu(self, prev_layer, layer_idx, layer_name):
        with tf.compat.v1.variable_scope(layer_name):
            # Obtain current weights
            W, b = self._weights(layer_idx, layer_name)
            # Convert the weights to tensor
            W = tf.constant(W, name="weights")
            b = tf.constant(b, name="bias")
            conv2d = tf.nn.conv2d(input=prev_layer,
                                  filters=W,
                                  strides=[1, 1, 1, 1],
                                  padding="SAME")
            # ReLU activation
            out = tf.nn.relu(conv2d + b)
        setattr(self, layer_name, out)

    def avgpool(self, prev_layer, layer_name):
        with tf.compat.v1.variable_scope(layer_name):
            # Avg pooling (replace the original max pooling for better performance stated in the paper)
            out = tf.nn.avg_pool(value=prev_layer,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME")

        setattr(self, layer_name, out)

    def load(self):
        self.conv2d_relu(self.input, 0, "block1_conv1")
        self.conv2d_relu(self.block1_conv1, 2, "block1_conv2")
        self.avgpool(self.block1_conv2, "block1_pool")
        self.conv2d_relu(self.block1_pool, 5, "block2_conv1")
        self.conv2d_relu(self.block2_conv1, 7, "block2_conv2")
        self.avgpool(self.block2_conv2, "block2_pool")
        self.conv2d_relu(self.block2_pool, 10, "block3_conv1")
        self.conv2d_relu(self.block3_conv1, 12, "block3_conv2")
        self.conv2d_relu(self.block3_conv2, 14, "block3_conv3")
        self.conv2d_relu(self.block3_conv3, 16, "block3_conv4")
        self.avgpool(self.block3_conv4, "block3_pool")
        self.conv2d_relu(self.block3_pool, 19, "block4_conv1")
        self.conv2d_relu(self.block4_conv1, 21, "block4_conv2")
        self.conv2d_relu(self.block4_conv2, 23, "block4_conv3")
        self.conv2d_relu(self.block4_conv3, 25, "block4_conv4")
        self.avgpool(self.block4_conv4, "block4_pool")
        self.conv2d_relu(self.block4_pool, 28, "block5_conv1")
        self.conv2d_relu(self.block5_conv1, 30, "block5_conv2")
        self.conv2d_relu(self.block5_conv2, 32, "block5_conv3")
        self.conv2d_relu(self.block5_conv3, 34, "block5_conv4")
        self.avgpool(self.block5_conv4, "block5_pool")