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
import tf_cnnvis

config = p.getParams()

if sys.argv[1] == "train" or sys.argv[1] == "visualize":
	dataset_path = sys.argv[2]
	loader = Loader(dataset_path, config, "bbox")

if sys.argv[1] == "train":
	arch = Arch(config)
	preds = arch.darknet()
	labels = tf.placeholder(tf.float32,[None,13,13,5,6])
	x = arch.getX()
	ls = losses.yolo_loss(preds, config, labels)
	tf.summary.scalar("loss", ls)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(ls)
	saver = tf.train.Saver()


if not os.path.exists(config["MODEL_SAVE_PATH"]):
		os.mkdir(config["MODEL_SAVE_PATH"])


if sys.argv[1] == "train":
	with tf.Session() as sess:
		train_writer = tf.summary.FileWriter( '/tmp/yolo-tf/train/train', sess.graph)
		merged = tf.summary.merge_all()

		sess.run(tf.global_variables_initializer())
		for i in range(config["EPOCH_SIZE"]):
			print ("epoch number :{}".format(i))
			for j in range(int(len(open(dataset_path+"train.txt","r").readlines())/config["BATCH_SIZE"])):	
				print ("doing stuff on {}th batch".format(j))
				images,labels_ = loader.next_batch(config["BATCH_SIZE"], print_img_files=False)
				summary = sess.run([merged,train_step], feed_dict={x:images,labels:labels_})
				# summary = sess.run([train_step], feed_dict={x:images,labels:labels_})
				ls_val = sess.run(ls, feed_dict={x:images,labels:labels_})
				print ("loss : {}".format(ls_val))
				train_writer.add_summary(summary[0], j)
			loader.set_batch_ptr(0)
			if i%100 == 0:
				save_path = saver.save(sess, config["MODEL_SAVE_PATH"]+"model_{}.ckpt".format(i))
				print ("Model at {} epoch saved at {}".format(i, save_path))
elif sys.argv[1] == "test":
	new_graph = tf.Graph()
	with tf.Session(graph=new_graph) as sess:
		tf.global_variables_initializer().run()
		layers = ['r', 'p', 'c']
		saver = tf.train.import_meta_graph(config["MODEL_SAVE_PATH"]+"model_{}.ckpt.meta".format(sys.argv[2]))
		checkpoint = tf.train.latest_checkpoint(config["MODEL_SAVE_PATH"])
		saver.restore(sess, checkpoint)
		print ("model restored!")
		img = data.utils.manip_image("dataset/images/carrot/carrot_21.jpg", config)
		img = img.reshape([1, config["IMAGE_H"], config["IMAGE_W"], 3])
		inp = tf.get_default_graph().get_tensor_by_name("input:0")
		out = tf.get_default_graph().get_tensor_by_name("predictions:0")
		tf_cnnvis.activation_visualization(sess_graph_path = None, value_feed_dict = {inp:img}, input_tensor=out, layers=layers, path_logdir='/tmp/tf_cnnvis', path_outdir='/tmp/')
		p = sess.run(out,feed_dict={inp:img})
		# p = np.reshape(p,[1, config["GRID_H"], config["GRID_W"], config["BOX"], 4 + 1 + config["CLASS"]])
		print (set(p[:,:,:,:,4].flatten()))
		
elif sys.argv[1] == "visualize":
	train_txt_path = os.path.join(sys.argv[2],"train.txt")
	_, t= loader.next_batch(batch_size=1, ptr=0, train_txt_path=train_txt_path, print_img_files=True)
	print (t.shape)	
	img = open(train_txt_path, "r").readlines()[0]
	img = os.path.join(os.getcwd(),img)
	img = img.strip("\n")
	if os.path.exists(img):
		img = data.utils.manip_image(img, config)
		draw.labels(t, img, config)
	else:
		print ("file doesn't exist")