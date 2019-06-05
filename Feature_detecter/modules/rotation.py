import cv2
import math
import numpy as np


class Rotation:
	def __init__(self, position, size=(5000, 5000)):
		self.p1, self.p2 = position
		self.size = size
	
	def find_lotpunkt(self, x1, y1, x2, y2):
		r = (-((x1 - self.p1) * (x1 - x2) + (y1 - self.p2) * (y1 - y2))) / ((x1 - x2) ** 2 + (y1 - y2) ** 2)
		l1 = int(x1 + r * (x1 - x2))
		l2 = int(y1 + r * (y1 - y2))
		return l1, l2
	
	def get_degree(self, l1, l2):
		grad = math.atan((self.p2 - l2) / (self.p1 - l1)) / (2 * math.pi) * 360
		if l2 < self.p2:
			if grad > 0:
				return grad
			else:
				return 180 + grad
		else:
			if grad > 0:
				return grad + 180
			else:
				return 360 + grad
	
	def mark_line(self, image):
		flag, b = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
		edges = cv2.Canny(b, 150, 200, 3, 5)
		lines = cv2.HoughLinesP(edges, 10, np.pi / 128, 290, minLineLength=400, maxLineGap=100)
		color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		if lines is not None:
			lines = lines.reshape((-1, 4))
			len_max = 0
			for x1, y1, x2, y2 in lines:
				lenght = math.sqrt(abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2)
				if len_max < lenght:
					len_max = lenght
					line = (x1, y1, x2, y2)
			cv2.line(color_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 10)
			l1, l2 = self.find_lotpunkt(line[0], line[1], line[2], line[3])
			cv2.line(color_image, (l1, l2), (self.p1, self.p2), (255, 0, 0), 10)
			cv2.line(color_image, (0, 2500), (2500, 2500), (0, 0, 255), 4)
			cv2.line(color_image, (5000, 2500), (2500, 2500), (0, 255, 0), 4)
			grad = self.get_degree(l1, l2)
		else:
			return color_image, None
		return color_image, grad
	
	@staticmethod
	def draw_map(data, size):
		image = np.zeros(size, dtype=np.uint8)
		for x, y in data:
			cv2.line(image, (x, y), (x, y), 255, 10)
		return image
	
	def main(self, pre_data, cal_rotation=False, last_rotation=None, rotation_counter=0):
		"""
		This Function combines a few steps, with are used in some other functions.
		Feel free to write your own main for other applications.
		:param pre_data: np.array() - Preprocessed data with the module from LidarFunctions.
		:param cal_rotation: bool() - Set True if you want to calculate the rotation compared to the last.
		:param last_rotation: None or int() or float() -  if not None, it will calculate the rotation_counter and
		return the new grad as last_rotation
		:param rotation_counter: float() or int() - it needs to be the last rotation_counter from the last processed scan if given
		:return: the colored image and the rotation or the image, the last_rotation and the rotation_counter
		"""
		image = self.draw_map(pre_data, self.size)
		image, rotation = self.mark_line(image)
		if cal_rotation:
			if rotation is not None:
				if last_rotation is not None:
					drotation = rotation - last_rotation
					rotation_counter += drotation
					last_rotation = rotation
					return image, last_rotation, rotation_counter
				else:
					last_rotation = rotation
					return image, last_rotation, rotation_counter
		return image, rotation, last_rotation
