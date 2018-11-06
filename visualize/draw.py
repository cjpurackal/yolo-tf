import matplotlib.pyplot as plt
import numpy as np
from os.path import join,exists
import data.utils as utils

def boxes(t, img, config, tf):
	if t is None:
		raise ValueError("Tensor cannot be None")
	# if img is None:
	# 	raise ValueError("Image cannot be None")
	# if exists(img):
	# 	raise ValueError("%s , couldn't find file"%img)
	# img = plt.imread(img)
	where = tf.not_equal(t[0,:,:,:,0],0)
	indices = tf.where(where)
	boxes = []
	for grid_x, grid_y, box_prior in indices:
		cx, cy, w, h = t[0, grid_x, grid_y, box_prior, 0:4]
		cx = cx * (self.config["GRID_W"]/self.config["IMAGE_W"])
		cy = cy * (self.config["GRID_H"]/self.config["IMAGE_H"])
		w = w * (self.config["GRID_W"]/self.config["IMAGE_W"])
		h = h * (self.config["GRID_H"]/self.config["IMAGE_H"])
		x0 = cx - w/2
		y0 = cy - h/2
		box = [x0, y0, w, h]
		boxes.append(box)

	print ("found %d boxes" % len(boxes))
	print (boxes)
