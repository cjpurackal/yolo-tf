
import tensorflow as tf
import numpy as np




def yolo_loss(y_pred, config, y_true):

	p_xy = np.reshape(config["ANCHORS"], [1, 1, 1, config["BOX"], 2])
	pred_xy = tf.sigmoid(y_pred[...,0:2])
	pred_wh = tf.exp(y_pred[...,2:4]) * p_xy
	objectness = y_pred[...,4]
	classes = y_pred[...,4:]

	true_xy = y_pred[..., :2]
	true_wh = y_pred[..., 2:4]

	#now going to compute the iou scores

	pred_wh_half = pred_wh / 2
	pred_mins = pred_xy - pred_wh_half
	pred_maxs = pred_xy + pred_wh_half
	pred_area = pred_wh[..., 0] * pred_wh[..., 1]


	true_wh_half = true_wh / 2
	true_mins = true_xy - true_wh_half
	true_maxs = true_xy + true_wh_half
	true_area = true_wh[..., 0] * true_wh[..., 1]

	intersection_cords_mins = tf.maximum(pred_mins, true_mins)
	intersection_cords_maxs = tf.minimum(pred_maxs, true_maxs)
	intersection_wh = tf.maximum(intersection_cords_maxs - intersection_cords_mins, 0)
	intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]

	total_area = true_area + pred_area - intersection_area

	iou = intersection_area / total_area

	
	
	# print "*********** anchors *********** \n\n"
	# print p_xy
	# print "\n\n"

	# print "*********** preds *********** \n\n"
	# print (pred_wh.eval())
	# print "\n\n"
	