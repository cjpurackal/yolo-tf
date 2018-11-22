import matplotlib.pyplot as plt
import numpy as np
from os.path import join,exists
import data.utils as utils

def labels(t, img, config):
	if t is None:
		raise ValueError("Tensor cannot be None")
	if img is None:
		raise ValueError("Image cannot be None")

	# if its a tensor
	# where = tf.not_equal(t[0,:,:,:,0],0)
	# indices = tf.where(where)
	indices = np.nonzero(t[0,:,:,:,4])
	# indices = [ind for ind in indices if ind.any()]
	indices = [[indices[0][ind], indices[1][ind], indices[2][ind]] for ind in range(len(indices[0]))]

	boxes = []
	for grid_x, grid_y, box_prior in indices:
		cx, cy, w, h = t[0, grid_x, grid_y, box_prior, 0:4]
		cx = (cx * config["IMAGE_W"])/ config["GRID_W"]
		cy = (cy * config["IMAGE_H"])/ config["GRID_H"]
		w = (w * config["IMAGE_W"])/ config["GRID_W"]
		h = (h *config["IMAGE_H"])/ config["GRID_H"]
		x0 = cx - w/2
		y0 = cy - h/2
		box = [x0, y0, w, h]
		boxes.append(box)

	draw_rects(img, boxes)


def draw_rects(img, boxes):
	import matplotlib.pyplot as plt
	import matplotlib.patches as patches
	import numpy as np

	fig,ax = plt.subplots(1)
	ax.imshow(img)
	for box in boxes:
		rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)

	plt.show()