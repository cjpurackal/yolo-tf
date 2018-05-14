class Bbox:
	def __init__(self, x, y, w, h):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
	def __getitem__(self,key):
		return getattr(self, key)
