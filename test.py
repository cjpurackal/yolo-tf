import tensorflow as tf
# tf.enable_eager_execution()

from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
import os
from network.darknet21 import Arch
import config.parameters as p
# from loss import losses
import data
from data.loader import Loader
from visualize import draw
import tf_cnnvis