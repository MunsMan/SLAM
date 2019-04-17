import numpy as np
from rplidar import RPLidar as rp
import time
from cv2 import line, add
from scipy.ndimage import rotate


class Lidar:
	
	def __init__(self, buffer):
		self.lidar = rp("/dev/tty.SLAB_USBtoUART")
		self.__buffer = buffer
	
	def start_lidar(self):
		self.lidar.connect()
		self.lidar.start_motor()
		info = self.lidar.get_info()
		time.sleep(2)
		return info
	
	def get_scan(self):
		"""
		:return: tuple(int(), list()) - Counter of Scans, list of result(quality, degree, distance)
		"""
		try:
			for i, scan in enumerate(self.lidar.iter_scans()):
				self.__buffer.put((i, scan))
				time.sleep(0.2)
		
		finally:
			self.stop()
	
	def stop(self):
		self.lidar.stop()
		self.lidar.stop_motor()
		self.lidar.disconnect()


class LidarFunktions:
	
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
	
	@staticmethod
	def draw_main_map(data, position, size, main_map, grad, color=200, thickness=2):
		zeros = np.zeros(size, np.uint8)
		for x, y in data:
			line(zeros, position, (x, y), color, thickness)
		zeros = rotate(zeros, grad)
		main_map = add(zeros, main_map)
		return main_map
	
	@staticmethod
	def draw_line_map(map, data, color=200, thickness=5):
		last_point = None
		for x, y in data:
			if last_point is None:
				last_point = (x, y)
			else:
				line(map, last_point, (x, y), color, thickness)
				last_point = (x, y)
		return map
	
	@staticmethod
	def draw_main_map_static(main_map, pre_data, position, size, color=200, thickness=2):
		zeros = np.zeros(size, np.uint8)
		for x, y in pre_data:
			line(zeros, position, (x, y), color, thickness)
		main_map = add(zeros, main_map)
		return main_map
