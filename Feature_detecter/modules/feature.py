import cv2
import numpy as np


class Feature:
	
	def __init__(self, image, scale=0, thresh=0.03, position=(5000, 5000)):
		self.image = image
		self.scale = scale
		self.thresh = thresh
		self.position = position
	
	def get_corners(self):
		image = cv2.resize(self.image,
		                   (self.image.shape[0] // self.scale, self.image.shape[0] // self.scale),
		                   interpolation=cv2.INTER_AREA)
		dst = cv2.cornerHarris(image, 2, 5, 0.05)
		dst = cv2.dilate(dst, None)
		image[dst > self.thresh * dst.max()] = [255]
		points = np.where(dst > self.thresh * dst.max())
		image = cv2.resize(image,
		                   (image.shape[0] * self.scale, image.shape[0] * self.scale),
		                   interpolation=cv2.INTER_CUBIC)
		return image, points


if __name__ == '__main__':
	import time
	
	img = cv2.imread("img773.jpg", 0)
	print(img.shape)
	st = time.time()
	img, points = Feature(img, 4).get_corners()
	print(points)
	print(time.time() - st)
	cv2.imshow("img", img)
	if cv2.waitKey(0) == ord('q'):
		cv2.destroyAllWindows()
		exit()
