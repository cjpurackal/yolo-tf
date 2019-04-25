import sys
import json	
import tensorflow as tf
import numpy as np

from yolo.data import Loader
from yolo.feature_extractors import Darknet21
from yolo.box_predictors import Yolo_box_predictor
from yolo.losses import yolo_loss
tf.enable_eager_execution()

config_path = sys.argv[1]
config = json.load(open(config_path))

pt = tf.Variable(np.random.rand(5,13,13,5,6), dtype=tf.float32)
tt = np.zeros((5,13,13,5,6), dtype=np.float32)
tt[:,5,5,2,:] = (.23, .11, .33, .33, 1, 1)
tt = tf.Variable(tt, dtype=tf.float32)
tb = tf.Variable(np.random.rand(5,1,1,1,10,4), dtype=tf.float32)

yl = yolo_loss(config, tt, pt)
print (yl)


#defining inputs
# inputs = tf.placeholder(dtype=tf.float32,shape=[None, config["input_size"], config["input_size"], 3], name="inputs")

#defining the model
# features = Darknet21(inputs).forward()
# boxes = Yolo_box_predictor(features, config).forward()
