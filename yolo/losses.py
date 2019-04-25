import tensorflow as tf
import numpy as np

def yolo_loss(config, prediction_tensor, truth_tensor):
	mask_shape = tf.shape(truth_tensor)[:4]
	conf_mask = tf.zeros(mask_shape)
	class_mask = tf.zeros(mask_shape)
	coord_mask = tf.zeros(mask_shape)

	class_wt = np.ones(len(config["labels"]), dtype='float32')

	cell_x = tf.reshape(tf.tile(tf.range(config["grid_size"], dtype=tf.float32),[config["grid_size"]]), [1, config["grid_size"], config["grid_size"], 1, 1])
	cell_y = tf.transpose(cell_x, [0, 2, 1, 3, 4])
	cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [config["batch_size"], 1, 1, config["boxes_per_cell"], 1])
	
	xy_pred_tensor = tf.sigmoid(prediction_tensor[:,:,:,:,:2]) + cell_grid
	wh_pred_tensor = tf.exp(prediction_tensor[:,:,:,:,2:4]) * np.reshape(config["anchors"], [1, 1, 1, config["boxes_per_cell"], 2])
	obj_conf_pred_tensor = tf.sigmoid(prediction_tensor[:,:,:,:,4])
	prob_dist_pred_tensor = prediction_tensor[:,:,:,:,5:]
	

	max_prob_pred_tensor = tf.argmax(prob_dist_pred_tensor, -1)
	max_prob_truth_tensor = tf.argmax(truth_tensor[..., 5:], -1)

	xy_truth_tensor = truth_tensor[:,:,:,:,0:2]
	wh_truth_tensor = truth_tensor[:,:,:,:,2:4]

	#calculating iou scores
	wh_half_truth_tensor = wh_truth_tensor/2
	true_mins = xy_truth_tensor - wh_half_truth_tensor
	true_maxs = xy_truth_tensor + wh_half_truth_tensor


	wh_half_pred_tensor = wh_pred_tensor/2
	pred_mins = xy_pred_tensor - wh_half_pred_tensor
	pred_maxs = xy_pred_tensor + wh_half_pred_tensor


	intersect_mins = tf.maximum(pred_mins, true_mins)
	intersect_maxs = tf.minimum(pred_maxs, true_maxs)
	intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0)
	intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

	true_areas = wh_truth_tensor[..., 0] * wh_truth_tensor[..., 1]
	pred_areas = wh_pred_tensor[..., 0] * wh_pred_tensor[..., 1]

	union_areas =  (pred_areas + true_areas) - intersect_areas
	iou_scores = tf.truediv(intersect_areas, union_areas)

	obj_conf_truth_tensor = truth_tensor[..., 4] * iou_scores

	conf_mask = conf_mask + tf.to_float(iou_scores < 0.6) * (1 - truth_tensor[..., 4]) * config["no_obj_scale"]
	class_mask = truth_tensor[..., 4] * tf.gather(class_wt, max_prob_pred_tensor) * config["class_scale"]       
	coord_mask = tf.expand_dims(truth_tensor[..., 4], axis=-1) * config["coord_scale"]

	nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
	nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
	nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

	loss_xy    = tf.reduce_sum(tf.square(xy_truth_tensor - xy_pred_tensor) * coord_mask) / (nb_coord_box + 1e-6) / 2.
	loss_wh    = tf.reduce_sum(tf.square(wh_truth_tensor - wh_pred_tensor) * coord_mask) / (nb_coord_box + 1e-6) / 2.
	loss_conf  = tf.reduce_sum(tf.square(obj_conf_truth_tensor- obj_conf_pred_tensor) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
	loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=max_prob_truth_tensor, logits=prob_dist_pred_tensor)
	loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
	loss = loss_xy + loss_wh + loss_conf + loss_class

	# debug = True
	# if debug:
	# 	nb_true_box = tf.reduce_sum(y_true[..., 4])
	# 	nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))
		
	# 	current_recall = nb_pred_box/(nb_true_box + 1e-6)
	# 	total_recall = tf.assign_add(total_recall, current_recall) 

	# 	loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
	# 	loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
	# 	loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
	# 	loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
	# 	loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
	# 	loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
	# 	loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)

	return loss









































	