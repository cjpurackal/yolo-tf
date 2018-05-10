import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from network.darknet import Arch
import config.parameters as p
from loss import yolo_loss

image = plt.imread("data/images/apple_73.jpg").astype(float).reshape(1, 416, 416, 3)
label = open("data/labels/apple_73.txt","r")

config = p.getParams()
arch = Arch(config)

preds = arch.darknet()
x = arch.getX()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	preds = preds.eval(feed_dict={x:image})
	yolo_loss.custom_loss(preds, config)



	