import tensorflow as tf
import numpy as np


class Arch:

	def __init__(self,config):
		self.config = config
		self.x = tf.placeholder(dtype=tf.float32,shape=[None, config["IMAGE_W"], config["IMAGE_H"], 3], name="input")

	def darknet(self):
		conv1 = tf.layers.conv2d(self.x, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv1 = tf.layers.batch_normalization(conv1)
		maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
		conv2 = tf.layers.conv2d(maxpool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv2 = tf.layers.batch_normalization(conv2)
		maxpool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
		conv3 = tf.layers.conv2d(maxpool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv3 = tf.layers.batch_normalization(conv3)
		conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=[1, 1], padding="same", activation=tf.nn.leaky_relu)
		conv4 = tf.layers.batch_normalization(conv4)
		conv5 = tf.layers.conv2d(conv4, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv5 = tf.layers.batch_normalization(conv5)
		maxpool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
		conv6 = tf.layers.conv2d(maxpool3, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv6 = tf.layers.batch_normalization(conv6)
		conv7 = tf.layers.conv2d(conv6, filters=128, kernel_size=[1, 1], padding="same", activation=tf.nn.leaky_relu)
		conv7 = tf.layers.batch_normalization(conv7)
		conv8 = tf.layers.conv2d(conv7, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv8 = tf.layers.batch_normalization(conv8)
		maxpool4 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
		conv9 = tf.layers.conv2d(maxpool4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv9 = tf.layers.batch_normalization(conv9)
		conv10 = tf.layers.conv2d(conv9, filters=256, kernel_size=[1, 1], padding="same", activation=tf.nn.leaky_relu)
		conv10 = tf.layers.batch_normalization(conv10)
		conv11 = tf.layers.conv2d(conv10, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv11 = tf.layers.batch_normalization(conv11)
		conv12 = tf.layers.conv2d(conv11, filters=256, kernel_size=[1, 1], padding="same", activation=tf.nn.leaky_relu)
		conv12 = tf.layers.batch_normalization(conv12)
		conv13 = tf.layers.conv2d(conv12, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv13 = tf.layers.batch_normalization(conv13)
		maxpool5 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
		conv14 = tf.layers.conv2d(maxpool5, filters=1024, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv14 = tf.layers.batch_normalization(conv14)
		conv15 = tf.layers.conv2d(conv14, filters=512, kernel_size=[1, 1], padding="same", activation=tf.nn.leaky_relu)
		conv15 = tf.layers.batch_normalization(conv15)
		conv16 = tf.layers.conv2d(conv15, filters=1024, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv16 = tf.layers.batch_normalization(conv16)
		conv17 = tf.layers.conv2d(conv16, filters=512, kernel_size=[1, 1], padding="same", activation=tf.nn.leaky_relu)
		conv17 = tf.layers.batch_normalization(conv17)
		conv18 = tf.layers.conv2d(conv17, filters=1024, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv18 = tf.layers.batch_normalization(conv18)
		conv19 = tf.layers.conv2d(conv18, filters=1024, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv19 = tf.layers.batch_normalization(conv19)
		conv20 = tf.layers.conv2d(conv19, filters=1024, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv20 = tf.layers.batch_normalization(conv20)
		reorg = tf.reshape(conv13,[self.config["BATCH_SIZE"],self.config["GRID_W"],self.config["GRID_H"],2048])
		route = tf.concat([reorg,conv20],3)
		conv21 = tf.layers.conv2d(route, filters=1024, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
		conv21 = tf.layers.batch_normalization(conv21)
		predictions = tf.layers.conv2d(conv21, filters=self.config["BOX"] * (4 + 1 + self.config["CLASS"]), kernel_size=[1, 1], padding="same", activation=tf.nn.leaky_relu)	
		predictions = tf.reshape(predictions, [predictions.shape[0], self.config["GRID_H"], self.config["GRID_W"], self.config["BOX"], 4 + 1 + self.config["CLASS"]], name="predictions")
		return predictions



	def getX(self):
		return self.x;










