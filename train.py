import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
from network.darknet21 import Arch
import config.parameters as p
from loss import losses
from data.loader import Loader

dataset_path = sys.argv[1]
config = p.getParams()
loader = Loader(dataset_path, config, "bbox")

arch = Arch(config)
preds = arch.darknet()
x = arch.getX()

with tf.Session() as sess:
	for i in range(int(len(open(dataset_path+"train.txt","r").readlines())/config["BATCH_SIZE"])):
		sess.run(tf.global_variables_initializer())
		print ("doing stuff on {}th batch".format(i))
		images,labels = loader.next_batch(config["BATCH_SIZE"])
		ls = losses.yolo_loss(preds, config, labels)
		train_step = tf.train.AdamOptimizer(1e-4).minimize(ls)
		sess.run(train_step, feed_dict={x:images})
		ls_val = sess.run(ls, feed_dict={x:images})
		print ("loss : {}".format(ls_val))


