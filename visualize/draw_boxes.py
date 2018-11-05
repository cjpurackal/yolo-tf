import matplotlib.pyplot as plt
import numpy as np
from os.path import join,exists
import data.utils as utils
import tensorflow as tf


def draw_box(t, img):
	if t is None:
		raise ValueError("Tensor cannot be None")
	# if img is None:
	# 	raise ValueError("Image cannot be None")
	# if exists(img):
	# 	raise ValueError("%s , couldn't find file"%img)
	# img = plt.imread(img)
	where = tf.not_equal(t[0,:,:,:,0],0)
	indices = tf.where(where)
	print (indices)