import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Lambda, Reshape
from models import Darknet21
from loss import yolo_loss
import config.parameters as p
import data
from data.loader import Loader

save_path = "/tmp/yolo-tf/"
saved_weights_path = sys.argv[1]
test_img = sys.argv[2]

config = p.getParams()


with tf.Session() as sess:
	saver = tf.train.import_meta_graph(saved_weights_path)
	checkpoint = tf.train.latest_checkpoint(save_path)
	saver.restore(sess, checkpoint)
	print ("model restored")
	img = data.utils.manip_image(test_img, config)
	img = img.reshape([1, config["IMAGE_H"], config["IMAGE_W"], 3])
	inp = tf.get_default_graph().get_tensor_by_name("input:0")
	# out = tf.get_default_graph().get_tensor_by_name("predictions:0")