#to do

import os
import numpy as np
from utils.Bbox import Bbox
import cv2



def get_data(config, root_dir="/dataset/"):

	instance_count = 0
	x_batch = np.zeros([config["BATCH_SIZE"], config["IMAGE_W"], config["IMAGE_H"], 3], np.float32)
	anchors = [Bbox(0, 0, config["ANCHORS"][2*i], config["ANCHORS"][2*i + 1]) for i in range(len(config["ANCHORS"])/2)]
	y_batch = np.zeros([config["BATCH_SIZE"], config["GRID_W"], config["GRID_H"], config["BOX"], 4+1+config["CLASS"]], np.float32)
	max_iou = -1
	best_prior = -1

	for file in os.listdir(os.getcwd() + root_dir + "images/"):

		labels_file = open(os.getcwd() + "/dataset/labels/" + file[:-3] + "txt").readlines()
		labels_all = [l+labels_file[i-1] for i, l in enumerate(labels_file) if i % 2 == 0]
		objs = util.convert_to_bbox(labels_all)

		image, objs = manip_image_and_label(os.getcwd() + root_dir + "images/" + file, objs, config)

		for obj in objs:						
			class_vector = np.zeros(config["CLASS"])
			class_vector[obj.cat] = 1

			center_x = .5 * (obj.xmin + obj.xmax)
			center_x = center_x / (config["IMAGE_W"]/config["GRID_W"])
			center_y = .5 * (obj.ymin + obj.ymax)
			center_y = center_y / (config["IMAGE_H"]/config["GRID_H"])

			center_w = (obj.xmax - obj.xmin) / (config["IMAGE_W"]/config["GRID_W"])
			center_h = (obj.ymax - obj.ymin) / (config["IMAGE_H"]/config["GRID_H"])

			# print "centerx = {} centery = {} centerw = {} centerh = {}".format(center_x, center_y, center_w, center_h)


			grid_x = int(np.floor(center_x))
			grid_y = int(np.floor(center_y))

			bbox = [center_x, center_y, center_w, center_h]

			box = Bbox(0, 0, center_w, center_h)

			for i in range(len(anchors)):
				iou = util.compute_iou(anchors[i], box)
				# print "iou is : {}".format(iou)

				if iou > max_iou:
					max_iou = iou
					best_prior = i
					# print "best iou is : {}".format(max_iou)

			y_batch[instance_count, grid_x, grid_y, best_prior, 0:4] = bbox
			y_batch[instance_count, grid_x, grid_y, best_prior, 4] = 1
			y_batch[instance_count, grid_x, grid_y, best_prior, 5:5+config["CLASS"]] = class_vector
			x_batch[instance_count] = image

		instance_count += 1

	return x_batch, y_batch



def manip_image_and_label(image_file, objs, config):
	image = cv2.imread(image_file)
	h, w, c = image.shape
	image = cv2.resize(image, (config["IMAGE_H"], config["IMAGE_W"]))
	for obj in objs:

		obj.xmin = int(obj.xmin * float(config['IMAGE_W']) / w)
		obj.xmin = max(min(obj.xmin, config['IMAGE_W']), 0)

		obj.xmax = int(obj.xmax * float(config['IMAGE_W']) / w)
		obj.xmax = max(min(obj.xmax, config['IMAGE_W']), 0)

		obj.ymin = int(obj.ymin * float(config['IMAGE_H']) / h)
		obj.ymin = max(min(obj.ymin, config['IMAGE_H']), 0)

	return image, objs




















