from tensorflow.keras.layers import Conv2D, Reshape

class Yolo_box_predictor:
	def __init__(self, features, config):
		self.features = features
		self.config = config

	def forward(self):
		out = Conv2D(self.config["boxes_per_cell"] * (4 + 1 + len(self.config["labels"])), 
				(1,1), strides=(1,1), 
				padding='same', 
				name='DetectionLayer', 
				kernel_initializer='lecun_normal')(self.features)
		return Reshape((self.config["grid_size"], self.config["grid_size"], self.config["boxes_per_cell"], 4 + 1 + len(self.config["labels"])), name="predictions")(out)
