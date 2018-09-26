import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2		
from network.darknet21 import Arch
import config.parameters as p
from loss import loss
from data.loader import Loader

dataset_path = "/home/christie/projects/hobby/yolo-tf/dataset/"
config = p.getParams()
loader = Loader(dataset_path, config, "bbox")

arch = Arch(config)
preds = arch.darknet()
x = arch.getX()

with tf.Session() as sess:
	for i in len(open(dataset_path+"train.txt","r").readlines())/config["BATCH_SIZE"]
		print ("doing stuff on {}th batch".format(i))
		loss = loss.yolo_loss(preds, config, labels)
		train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
		sess.run(tf.global_variables_initializer())
		sess.run(train_step, feed_dict={x:images})


