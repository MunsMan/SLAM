from .scan import LidarFunctions
import cv2
import numpy as np
import math
from .image_queue import ImageQueue


class Motion:
	
	def __init__(self, size):
		self.size = size
		self.lf = LidarFunctions()
		self.queue = ImageQueue(2, 1)
	
	@staticmethod
	def find_lines(image):
		flag, b = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
		edges = cv2.Canny(b, 150, 200, 3, 5)
		lines = cv2.HoughLinesP(edges, 10, np.pi / 128, 290, minLineLength=400, maxLineGap=100)
		if lines is not None:
			if len(lines) >= 3:
				lines = lines.reshape((-1, 4))
				sorted = sorted(
					[[math.sqrt(abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2), x1, y1, x2, y2] for x1, y1, x2, y2 in lines])
				return sorted[-4:-1]
			else:
				return 0
		else:
			return 0
	
	@staticmethod
	def draw(image, lines, color=255, thickness=8):
		for x1, y1, x2, y2 in lines:
			cv2.line(image, (x1, y1), (x2, y2), color, thickness)
		return image
	
	def main(self, predata, last_image):
		new_image = self.lf.draw_point_cloud(predata, self.size)
		new_lines = self.find_lines(new_image)
		if self.queue.len == 0:
			return 0
		base_image = self.queue.get_image()
		base_lines = self.find_lines(base_image)
		base_image = self.draw(base_image, new_lines)
		new_image = self.draw(new_image, new_lines)
		image = np.hstack((base_image, new_image))
		cv2.imshow("Image", image)
		cv2.waitKey(0)
		cv2.destroyWindow("Image")
