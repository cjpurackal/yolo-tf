import tensorflow as tf
import numpy as np



def custom_loss(y_pred,config):
	mask_shape = tf.shape(y_pred)[:4]

	cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(config["GRID_W"]), [config["GRID_H"]]), (1, config["GRID_H"], config["GRID_W"], 1, 1)))
	cell_y = tf.transpose(cell_x, (0,2,1,3,4))
	cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [config["BATCH_SIZE"], 1, 1, 5, 1])

	coord_mask = tf.zeros(mask_shape)
	conf_mask  = tf.zeros(mask_shape)
	class_mask = tf.zeros(mask_shape)

	seen = tf.Variable(0.)
	total_recall = tf.Variable(0.)


	pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
	pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])

	pred_box_conf = tf.sigmoid(y_pred[..., 4])
    pred_box_class = y_pred[..., 5:]


