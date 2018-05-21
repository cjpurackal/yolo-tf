import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from network.darknet import Arch
import config.parameters as p
from loss import loss
import cv2
from utils import datamaker			




config = p.getParams()
images, labels = datamaker.get_data(config)


arch = Arch(config)

preds = arch.darknet()
x = arch.getX()

with tf.Session() as sess:
	loss = loss.yolo_loss(preds, config, labels)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
	sess.run(tf.global_variables_initializer())
	sess.run(train_step, feed_dict={x:images})


