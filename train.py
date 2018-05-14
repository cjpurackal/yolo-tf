# import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from network.darknet import Arch
import config.parameters as p
from loss import loss
import cv2
from utils import datamaker

# image = plt.imread("dataset/images/apple_73.jpg").astype(float).reshape(1, 416, 416, 3)
# image = cv2.imread("dataset/images/apple_73.jpg")
# image = cv2.resize(image,(416, 416))
# image = image / 255.
# image = image[:,:,::-1]
# image = np.expand_dims(image, 0)

# labels = open("dataset/labels/apple_73.txt","r").readline().replace("\n","").split(" ")[1:]
# labels = [ float(x) for x in labels]
# labels = np.array(labels)
# labels = labels[np.newaxis,:]


config = p.getParams()
# arch = Arch(config)

# preds = arch.darknet()
# x = arch.getX()

# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	preds = preds.eval(feed_dict={x:image})
# 	loss.yolo_loss(preds, config, labels)


datamaker.getlabelsinbigfattensorfrom(config)