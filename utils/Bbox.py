class Bbox:
	def __init__(self, xmin, ymin, xmax, ymax, cat = None):
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		self.cat = cat
	def __getitem__(self,key):
		return getattr(self, key)
