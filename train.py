import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Lambda, Reshape
from models import Darknet21
from loss import yolo_loss
import config.parameters as p
import data
from data.loader import Loader


config = p.getParams()
image_width = config["IMAGE_W"]
image_height = config["IMAGE_H"]
max_box_per_image = config["BOX"]
dataset_path = sys.argv[1]
loader = Loader(dataset_path, config, "bbox")

labels_ = tf.placeholder(tf.float32,[None, 13, 13, 5, 6])
b_batch_ = tf.placeholder(tf.float32, [None, 1, 1, 1, config["TRUE_BOX_BUFFER"], 4])

feature_extractor = Darknet21(config)
inputs = feature_extractor.get_input()
features = feature_extractor.forward()
outputs = Conv2D(config["BOX"] * (4 + 1 + config["CLASS"]), 
				(1,1), strides=(1,1), 
				padding='same', 
				name='DetectionLayer', 
				kernel_initializer='lecun_normal')(features)
outputs = Reshape((config["GRID_H"], config["GRID_W"], config["BOX"], 4 + 1 + config["CLASS"]))(outputs)
loss = yolo_loss.custom_loss(config, labels_, b_batch_, outputs)
tf.summary.scalar("loss", loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
saver = tf.train.Saver()


# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	images, b_batch, labels = loader.next_batch(config["BATCH_SIZE"], print_img_files=False)
# 	sess.run(loss, feed_dict={inputs:images, b_batch_:b_batch, labels_:labels})

with tf.Session() as sess:
  train_writer = tf.summary.FileWriter( '/tmp/yolo-tf/train/train', sess.graph)
  merged = tf.summary.merge_all()
  
  sess.run(tf.global_variables_initializer())
  for i in range(config["EPOCH_SIZE"]):
    print ("epoch number :{}".format(i))
    for j in range(int(len(open(dataset_path+"train.txt","r").readlines())/config["BATCH_SIZE"])):  
        print ("doing stuff on {}th batch".format(j))
        images, b_batch, labels = loader.next_batch(config["BATCH_SIZE"], print_img_files=False)
        summary = sess.run([merged, train_step], feed_dict={inputs:images, b_batch_:b_batch, labels_:labels})
        l = sess.run(loss, feed_dict={inputs:images, b_batch_:b_batch, labels_:labels})
        if np.isnan(l):
          exit(0)
        train_writer.add_summary(summary[0], j)
    loader.set_batch_ptr(0)
    if i%100 == 0:
      save_path = saver.save(sess, config["MODEL_SAVE_PATH"]+"model_{}.ckpt".format(i))
      print ("Model at {} epoch saved at {}".format(i, save_path))










