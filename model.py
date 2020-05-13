import os
import numpy as np
import tensorflow as tf
import cv2
from vgg19 import VGG
from utils import *
from os.path import isfile, join
import argparse

from tensorflow.compat.v1 import variable_scope, get_variable, Session, global_variables_initializer, train, disable_eager_execution
tf.compat.v1.disable_eager_execution()

class Image(object):
    def __init__(self, content_filepath, style_filepath, img_h, img_w, lr, frame_idx, prev_frame):
        self.img_height = img_h
        self.img_width = img_w
        
        self.content_img = get_resized_image(content_filepath, self.img_width, self.img_height)
        self.style_img = get_resized_image(style_filepath, self.img_width, self.img_height)
        self.initial_img = generate_noise_image(self.content_img, self.img_width, self.img_height)
        self.frame_idx = frame_idx
        # prev_frame = None if the image is the first frame
        self.prev_frame = prev_frame

        # Layers in which we compute the content/style loss
        self.content_layer = "block5_conv2"
        self.style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

        # Hyperparameters alpha and beta
        self.content_w = 1e-4
        self.style_w = 1
        self.temporal_w = 0.5

        # Initialize the weights for all style layers; this hyperparameter can be tuned
        # General idea: deeper layers are more important
        self.style_layer_w = [0.5 * i + 0.5 for i in range(5)]

        self.lr = lr
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")  # Global step
        print("Current frame: ", self.frame_idx)


    def create_input(self):
        """
        Initialize input tensor
        """
        with tf.compat.v1.variable_scope("input", reuse=tf.AUTO_REUSE):
            self.input_img = tf.compat.v1.get_variable("in_img", shape=([1, self.img_height, self.img_width, 3]), dtype=tf.float32, initializer=tf.zeros_initializer())

    def load_vgg(self):
        self.vgg = VGG(self.input_img)
        self.vgg.load()
        # mean-centering
        self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

    # This section contains the loss function and four helper functions.
    def _content_loss(self, img, content):
        self.content_loss = tf.reduce_sum(tf.square(img - content))
        
    def _gram_matrix(self, img, area, num_channels):
        """
        Compute the gram matrix G for an image tensor
        :param img: the feature map of an image, of shape (h, w, num_channels)
        :param area: h * w for some image
        :param num_channels: the number of channels in some image feature map
        """
        mat = tf.reshape(img, (area, num_channels))
        gram = tf.matmul(tf.transpose(mat), mat)
        return gram

    def _layer_style_loss(self, style, img):
        """
        Compute the style loss in a single layer
        :param img: the input image
        :param style: the style image
        """
        num_channels = style.shape[3]
        area = style.shape[1] * style.shape[2]

        gram_style = self._gram_matrix(style, area, num_channels)
        gram_img = self._gram_matrix(img, area, num_channels)

        return tf.reduce_sum(tf.square(gram_img - gram_style)) / ((2 * area * num_channels) ** 2)

    def _style_loss(self, style_maps):
        """
        Compute the total style loss across all layers
        :param style_maps: a set of all feature maps for the style image
        """
        # We use self.style_layers specified above to compute the total style loss
        num_layers = len(style_maps) # Should be 5

        unweighted_loss = [self._layer_style_loss(style_maps[i], getattr(self.vgg, self.style_layers[i]))
             for i in range(num_layers)]

        self.style_loss = sum(self.style_layer_w[i] * unweighted_loss[i] for i in range(num_layers))

    # def warp_image(self, img, flow):
    #     h, w = flow.shape[:2]
    #     flow = -flow
    #     flow[:,:,0] += np.arange(w)
    #     flow[:,:,1] += np.arange(h)[:,np.newaxis]
    #     res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    #     return res

    # def make_opt_flow(self, prev, nxt):
    #     print(prev.shape)
    #     print(nxt.shape)
    #     prev_gray = cv2.cvtColor(np.float32(prev[0,:,:,:]), cv2.COLOR_BGR2GRAY)
    #     nxt_gray = cv2.cvtColor(np.float32(nxt[0, :, :, :]), cv2.COLOR_BGR2GRAY)
    #     flow = cv2.calcOpticalFlowFarneback(prev_gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #     return flow

    # def get_prev_warped_frame(self, frame):
    #     prev_frame = self.prev_frame
    #     # backwards flow: current frame -> previous frame
    #     flow = self.make_opt_flow(prev_frame, frame)
    #     warped_img = self.warp_image(prev_frame, flow).astype(np.float32)
    #     # img = preprocess(warped_img)
    #     return warped_img

    # #flow1 is back, flow2 is forward
    # def get_flow_weights(self, flow1, flow2): 
    #     xSize = flow1.shape[1]
    #     ySize = flow1.shape[0]
    #     reliable = 255 * np.ones((ySize, xSize))

    #     size = xSize * ySize

    #     x_kernel = [[-0.5, -0.5, -0.5],[0., 0., 0.],[0.5, 0.5, 0.5]]
    #     x_kernel = np.array(x_kernel, np.float32)
    #     y_kernel = [[-0.5, 0., 0.5],[-0.5, 0., 0.5],[-0.5, 0., 0.5]]
    #     y_kernel = np.array(y_kernel, np.float32)
        
    #     flow_x_dx = cv2.filter2D(flow1[:,:,0],-1,x_kernel)
    #     flow_x_dy = cv2.filter2D(flow1[:,:,0],-1,y_kernel)
    #     dx = np.stack((flow_x_dx, flow_x_dy), axis = -1)

    #     flow_y_dx = cv2.filter2D(flow1[:,:,0],-1,x_kernel)
    #     flow_y_dy = cv2.filter2D(flow1[:,:,0],-1,y_kernel)
    #     dy = np.stack((flow_y_dx, flow_y_dy), axis = -1)

    #     motionEdge = np.zeros((ySize,xSize))

    #     for i in range(ySize):
    #         for j in range(xSize): 
    #             motionEdge[i,j] += dy[i,j,0]*dy[i,j,0]
    #             motionEdge[i,j] += dy[i,j,1]*dy[i,j,1]
    #             motionEdge[i,j] += dx[i,j,0]*dx[i,j,0]
    #             motionEdge[i,j] += dx[i,j,1]*dx[i,j,1]
            

    #     for ax in range(xSize):
    #         for ay in range(ySize): 
    #             bx = ax + flow1[ay, ax, 0]
    #             by = ay + flow1[ay, ax, 1]    

    #             x1 = int(bx)
    #             y1 = int(by)
    #             x2 = x1 + 1
    #             y2 = y1 + 1
                
    #             if x1 < 0 or x2 >= xSize or y1 < 0 or y2 >= ySize:
    #                 reliable[ay, ax] = 0.0
    #                 continue 
                
    #             alphaX = bx - x1 
    #             alphaY = by - y1

    #             a = (1.0-alphaX) * flow2[y1, x1, 0] + alphaX * flow2[y1, x2, 0]
    #             b = (1.0-alphaX) * flow2[y2, x1, 0] + alphaX * flow2[y2, x2, 0]
                
    #             u = (1.0 - alphaY) * a + alphaY * b
                
    #             a = (1.0-alphaX) * flow2[y1, x1, 1] + alphaX * flow2[y1, x2, 1]
    #             b = (1.0-alphaX) * flow2[y2, x1, 1] + alphaX * flow2[y2, x2, 1]
                
    #             v = (1.0 - alphaY) * a + alphaY * b
    #             cx = bx + u
    #             cy = by + v
    #             u2 = flow1[ay,ax,0]
    #             v2 = flow1[ay,ax,1]
                
    #             if ((cx-ax) * (cx-ax) + (cy-ay) * (cy-ay)) >= 0.01 * (u2*u2 + v2*v2 + u*u + v*v) + 0.5: 
    #                 # Set to a negative value so that when smoothing is applied the smoothing goes "to the outside".
    #                 # Afterwards, we clip values below 0.
    #                 reliable[ay, ax] = -255.0
    #                 continue
                
    #             if motionEdge[ay, ax] > 0.01 * (u2*u2 + v2*v2) + 0.002: 
    #                 reliable[ay, ax] = MOTION_BOUNDARIE_VALUE
    #                 continue
                
    #     #need to apply smoothing to reliable matrix
    #     reliable = cv2.GaussianBlur(reliable,(3,3),0)
    #     reliable = np.clip(reliable, 0.0, 255.0)    
    #     return reliable
    
    # def _temporal_loss(self, nxt, warped_prev, c):
    #     D = tf.size(nxt, out_type = tf.float32)
    #     loss = (1. / D) * tf.reduce_sum(tf.multiply(c, tf.squared_difference(nxt, warped_prev)))
    #     loss = tf.cast(loss, tf.float32)
    #     return loss

    # def sum_short_temporal_loss(self, frame):
    #     prev_frame = self.prev_frame
    #     if self.frame_idx == 0:
    #         return 0
    #     forward_flow = self.make_opt_flow(prev_frame, frame)
    #     backward_flow = self.make_opt_flow(frame, prev_frame)
    #     wp = self.warp_image(prev_frame)
    #     c = self.get_flow_weights(backward_flow, forward_flow)
    #     loss = self._temporal_loss(frame, wp, c)
    #     return loss

    def loss(self):
        """
        Compute the total loss of the model
        """
        with tf.compat.v1.variable_scope("loss", reuse=tf.AUTO_REUSE):
            # Content loss
            with tf.compat.v1.Session() as sess:
                sess.run(self.input_img.assign(self.content_img))
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            self._content_loss(content_img_content, gen_img_content)
            # # Temporal loss
            # self.temporal_loss = self.sum_short_temporal_loss(self.input_img)

            # Style loss
            with tf.compat.v1.Session() as sess:
                sess.run(self.input_img.assign(self.style_img))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])                              
            self._style_loss(style_layers)

            # Total loss; update to self.total_loss for future optimization
            self.total_loss = self.content_w * self.content_loss + self.style_w * self.style_loss

    def optimize(self):
        self.optimizer = train.AdamOptimizer(self.lr).minimize(self.total_loss, global_step=self.gstep)

    def build(self):
        self.create_input()
        self.load_vgg()
        self.loss()
        self.optimize()
    
    def train(self, n_iters=50):
        with tf.compat.v1.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.input_img.assign(self.initial_img))

            initial_step = self.gstep.eval()

            for epoch in range(initial_step, n_iters):
                
                sess.run(self.optimizer)
                if epoch == (n_iters - 1):
                    gen_image, total_loss = sess.run([self.input_img, self.total_loss])
                    # Inverse mean-centering
                    gen_image += self.vgg.mean_pixels 

                    print("Epoch: ", (epoch + 1))
                    print("Loss: ", total_loss)

                    filepath = "./output/frame_%d.png" % self.frame_idx
                    save_image(filepath, gen_image)
                    return gen_image