# import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from network.darknet import Arch
import config.parameters as p
from loss import loss
import cv2
from utils import datamaker			




config = p.getParams()
images, labels = datamaker.get_data(config)
# arch = Arch(config)

# preds = arch.darknet()
# x = arch.getX()

# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	preds = preds.eval(feed_dict={x:image})
# 	loss.yolo_loss(preds, config, labels)
