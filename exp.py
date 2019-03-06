import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras as k
import numpy as np


inp = tf.placeholder(dtype=tf.float32, shape=[None, 5, 5, 3])
out = tf.placeholder(dtype=tf.float32, shape=[None,2])

foo = Model(inp, out)


sample_in = np.random.rand(2,5,5,3)
sample_out = np.random.rand(2,2)
with tf.Session() as sess:
	sess.run(inp, feed_dict={inp:sample})