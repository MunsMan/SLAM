import numpy as np
import cv2


class Matches:
	def __init__(self, img1, points1, img2, points2, color=(255, 0, 0)):
		self.img1 = img1
		self.img2 = img2
		self.p1 = points1
		self.p2 = points2
		self.c = color
	
	def match_drawer(self):
		img3 = np.hstack((self.img1, self.img2))
		self.p2 = self.transform_p2()
		if self.p1.shape == self.p2.shape:
			for i in range(self.p1.shape[0] - 1):
				img3 = cv2.line(img3,
				                (int(self.p1[i][0]), int(self.p1[i][1])),
				                (int(self.p2[i][0]), int(self.p2[i][1])),
				                self.c)
			
			return img3
		else:
			print("Shapes don't match.")
			print(self.p1.shape, self.p2.shape)
	
	@staticmethod
	def move_points(points, move):
		return points + move
	
	def transform_p2(self):
		mx, my = self.img1.shape[0], self.img1.shape[1]
		px = self.move_points(self.p2[:, 0], mx)
		py = self.p2[:, 1]
		p2 = np.stack((px[:], py[:]), axis=1)
		return p2
