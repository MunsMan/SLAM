import numpy as np
import cv2
from imutils import rotate


class LidarFunctions:
	
	@staticmethod
	def get_coords(d, r):
		"""
		:param d: float() - degree
		:param r: float() - radius, distance in mm
		:return: tuple() - as coordinates
		"""
		x = int(round(np.cos(d * np.pi / 180) * r, 1))
		y = int(round(np.sin(d * np.pi / 180) * r, 1))
		return x, y
	
	@staticmethod
	def map_coords(points, position):
		"""
		function to transform points into map
		:param points: tuple() - points
		:param position: tuple() - position
		:return: tuple() - transformed points
		"""
		return points[0] + position[0], points[1] + position[1]
	
	def prepare_data(self, data, position):
		xy_data = []
		for q, d, r in data:
			x, y = self.get_coords(d, r)
			x, y = self.map_coords((x, y), position)
			xy_data.append((x, y))
		return np.array(xy_data)
	
	def prepare_data_to_3D(self, data, position):
		xy_data = []
		for q, d, r in data:
			x, y = self.get_coords(d, r)
			x, y = self.map_coords((x, y), position)
			xy_data.append((x, y, 0))
		return np.array(xy_data)
	
	@staticmethod
	def just_main_map(data, position, size, color=200, thickness=2):
		map = np.zeros(size, np.uint8)
		for x, y in data:
			cv2.line(map, position, (x, y), color, thickness)
		return map
	
	@staticmethod
	def draw_main_map(data, position, size, main_map, color=200, thickness=2):
		zeros = np.zeros(size, np.uint8)
		for x, y in data:
			cv2.line(zeros, position, (x, y), color, thickness)
		main_map = cv2.add(zeros, main_map)
		return main_map
	
	@staticmethod
	def draw_line_map(map, data, color=200, thickness=5):
		last_point = None
		for x, y in data:
			if last_point is None:
				last_point = (x, y)
			else:
				cv2.line(map, last_point, (x, y), color, thickness)
				last_point = (x, y)
		return map
	
	@staticmethod
	def draw_and_add_main_map(main_map, pre_data, position, rotation, size, thresh, i, color=200, thickness=2):
		zeros = np.zeros(size, np.uint8)
		for x, y in pre_data:
			cv2.line(zeros, position, (x, y), color, thickness)
		zeros = rotate(zeros, rotation)
		match = LidarFunctions.review_match(main_map, zeros)
		print("Review:", match)
		if 80 < i and match < thresh:
			return main_map
		main_map = cv2.add(main_map, zeros)
		return main_map
	
	@staticmethod
	def draw_point_cloud(pre_data, size, color=255, thickness=8):
		zeros = np.zeros(size, np.uint8)
		for x, y in pre_data:
			cv2.line(zeros, (x, y), (x, y), color, thickness)
		return zeros
	
	@staticmethod
	def draw_main_map_review(main_map, pre_data, position, rotation, size, thresh, i, color=255, thickness=2):
		zeros = np.zeros(size, np.uint8)
		for x, y in pre_data:
			cv2.line(zeros, position, (x, y), color, thickness)
		zeros = rotate(zeros, rotation)
		match = LidarFunctions.review_match(main_map, zeros)
		print("Review:", match)
		if 80 < i and match < thresh:
			return None, main_map
		return True, zeros
	
	@staticmethod
	def review_match(img1, img2):
		thresh_img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY)[1]
		thresh_img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY)[1]
		obj_count2 = np.count_nonzero(thresh_img2)
		array = np.logical_and(thresh_img1, thresh_img2)
		matches = np.count_nonzero(array)
		return matches / obj_count2
	
	@staticmethod
	def evaluate_rotation(img1, img2, rotation):
		assert img1.shape == img2.shape
		w1 = np.sum(np.isin(img1, np.max(img1)))
		w2 = np.sum(np.isin(img2, np.max(img2)))
		dw = abs(w1 - w2)
		print(np.max(img1))
		print(np.count_nonzero(np.isin(img1, np.max(img1))))
		return np.logical_and(np.isin(img1, np.max(img1)), np.isin(img2, np.max(img2)))
