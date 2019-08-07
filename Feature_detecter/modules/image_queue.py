import numpy as np
from cv2 import add


class ImageQueue:
	def __init__(self, queue_len, img_depth_code):
		self.queue_len = queue_len
		self._store = []
		self.len = len(self._store)
		self.depth = self.cal_depth(img_depth_code)
	
	@staticmethod
	def cal_depth(depth):
		if depth == 0:
			return None
		
		elif depth == 1:
			return np.bool
		
		elif depth == 2:
			return np.uint8
		else:
			return None
	
	def put(self, image):
		self._store.append(image)
		self.len = len(self._store)
		if self.len > self.queue_len:
			self._store.pop(0)
			self.len = len(self._store)
	
	def _create_image(self):
		image = self._store[0]
		for i in range(self.len - 1):
			image2 = self._store[i + 1]
			image = add(image, image2)
		return np.array(image, self.depth)
	
	def place(self, image, add=True):
		self.put(image)
		if add:
			added_image = self._create_image()
			return added_image
	
	def get_image(self):
		return self._create_image()


if __name__ == '__main__':
	image_1 = np.array([[1, 0], [1, 0]])
	image_2 = np.array([[0, 1], [1, 0]])
	image_3 = np.array([[0, 1], [0, 1]])
	image_4 = np.array([[1, 0], [0, 1]])
	image_5 = np.array([[0, 0], [0, 0]])
	
	Q = ImageQueue(3, 1)
	print(Q.place(image_1))
	print(Q.place(image_2))
	print(Q.place(image_3))
	print(Q.place(image_4))
	print(Q.place(image_5))
	print(Q.place(image_5))
	print(Q._store)
	print(Q.place(image_5))
