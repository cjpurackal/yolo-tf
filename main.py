import tensorflow as tf
# tf.enable_eager_execution()

from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
import os
from network.darknet21 import Arch
import config.parameters as p
from loss import losses
import data
from data.loader import Loader
from visualize import draw


config = p.getParams()

if sys.argv[1] == "train" or "visualize":
	dataset_path = sys.argv[2]
	loader = Loader(dataset_path, config, "bbox")

if sys.argv[1] == "train":
	arch = Arch(config)
	preds = arch.darknet()
	labels = tf.placeholder(tf.float32,[None,13,13,5,6])
	x = arch.getX()
	saver = tf.train.Saver()
	ls = losses.yolo_loss(preds, config, labels)
	tf.summary.scalar("loss", ls)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(ls)

if not os.path.exists(config["MODEL_SAVE_PATH"]):
		os.mkdir(config["MODEL_SAVE_PATH"])

with tf.Session() as sess:
	if sys.argv[1] == "train":
		train_writer = tf.summary.FileWriter( '/tmp/yolo-tf/train/train', sess.graph)
		sess.run(tf.global_variables_initializer())
		for i in range(config["EPOCH_SIZE"]):
			print ("epoch number :{}".format(i))
			for j in range(int(len(open(dataset_path+"train.txt","r").readlines())/config["BATCH_SIZE"])):	
				print ("doing stuff on {}th batch".format(j))
				images,labels_ = loader.next_batch(config["BATCH_SIZE"], print_img_files=False)
				
				summary = sess.run([train_step], feed_dict={x:images,labels:labels_})
				ls_val = sess.run(ls, feed_dict={x:images,labels:labels_})
				print ("loss : {}".format(ls_val))

			merged = tf.summary.merge_all()
			summary = sess.run(merged)
			train_writer.add_summary(summary[0], i)				
			loader.set_batch_ptr(0)
			if i%100 == 0:
				save_path = saver.save(sess, config["MODEL_SAVE_PATH"]+"model_{}.ckpt".format(i))
				print ("Model at {} epoch saved at {}".format(i, save_path))
	elif sys.argv[1] == "test":
		saver = tf.train.import_meta_graph(config["MODEL_SAVE_PATH"]+"model_900.ckpt.meta")
		saver.restore(sess, config["MODEL_SAVE_PATH"]+"model_900.ckpt")
		print ("model restored!")
		img = data.utils.manip_image(sys.argv[2], config)
		print (type(img))
		img = img.reshape([1, config["IMAGE_H"], config["IMAGE_W"], 3])
		print (img.shape)
		inp = tf.get_default_graph().get_operation_by_name('Placeholder:0').outputs[0]
		out = tf.get_default_graph().get_operation_by_name('Placeholder_1:0').outputs[0]
		# for v in tf.get_default_graph().get_collection("variables"):
		# 	print (v)
		# 	input()
		# for x in tf.get_default_graph().get_operations():
		# 	if x.type == "Placeholder":
		# 		print(x)		
		# 		input()
		# input()
		# for v in tf.get_default_graph().get_collection("trainable_variables"):
		# 	print (v)
		# 	input()
		p = sess.run(out,feed_dict={inp:img})
		p = np.reshape(p,[1, config["GRID_H"], config["GRID_W"], config["BOX"], 4 + 1 + config["CLASS"]])
		print (p.shape)
	elif sys.argv[1] == "visualize":
		train_txt_path = os.path.join("dataset","train.txt")
		_, t= loader.next_batch(batch_size=1, ptr=0, train_txt_path=train_txt_path, print_img_files=True)	
		img = open(train_txt_path, "r").readlines()[0]
		img = os.path.join(os.getcwd(),img)
		img = img.strip("\n")
		if os.path.exists(img):
			img = data.utils.manip_image(img, config)
			draw.labels(t, img, config)
		else:
			print ("file doesn't exist")