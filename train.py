import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from network.darknet21 import Arch
import config.parameters as p
from loss import loss
import cv2
from utils import datamaker			
from data.loader import Loader


config = p.getParams()
loader = Loader("/home/christie/projects/hobby/yolo-tf/dataset/", config, "bbox")

for _ in range(8):
	images, labels = loader.next_batch(2)

# arch = Arch(config)

# preds = arch.darknet()
# x = arch.getX()

# with tf.Session() as sess:
# 	loss = loss.yolo_loss(preds, config, labels)
# 	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
# 	sess.run(tf.global_variables_initializer())
# 	sess.run(train_step, feed_dict={x:images})


