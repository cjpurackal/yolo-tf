#to do

import os
import numpy as np
from Bbox import Bbox
import util




def getlabelsinbigfattensorform(config, labels_dir="/dataset/labels/"):

	instance_count = 0
	anchors = [Bbox(0, 0, config["ANCHORS"][2*i], config["ANCHORS"][2*i + 1]) for i in range(len(config["ANCHORS"])/2)]
	y_batch = np.zeros([config["BATCH_SIZE"], config["GRID_W"], config["GRID_H"], config["BOX"], 4+1+config["CLASS"]])
	max_iou = -1
	best_prior = -1

	for file in os.listdir(os.getcwd() + labels_dir):
		labels = open(os.getcwd() + labels_dir + file).read().split("\n")

		if len(labels) > 2:
			print "yet to do"
		else:
			label = [float(x) for x in labels[0].split(" ")]
			
			class_vector = np.zeros(config["CLASS"])
			class_vector[label[0]] = 1

			grid_x = np.floor(label[1] / config["GRID_W"])
			grid_y = np.floor(label[2] / config["GRID_H"])
			grid_w = label[3] / config["GRID_W"]
			grid_h = label[4] / config["GRID_H"]
			bbox = [grid_x, grid_y, grid_w, grid_h]
			box = [0, 0, grid_w, grid_h]
			for i in range(len(config["ANCHORS"])/2):
				iou = util.compute_iou(anchors[i], box)

				if iou > max_iou:
					max_iou = iou
					best_prior = i

		y_batch[instance_count, grid_x, grid_y, best_prior, 0:4] = bbox
		y_batch[instance_count, grid_x, grid_y, best_prior, 4] = 1
		y_batch[instance_count, grid_x, grid_y, best_prior, 5:5+len(config["CLASS"])] = class_vector












