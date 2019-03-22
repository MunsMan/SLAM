import numpy as np
from rplidar import RPLidar as rp
import time
from cv2 import line


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
		:param rotaions: int() - Defines, how many rotaions get returned
		:return: tuple(int(), list()) - Counter of Scans, list of result(quality, degree, distance
		)
		"""
		try:
			for i, scan in enumerate(self.lidar.iter_scans()):
				self.__buffer.put((i, scan))
		
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
		:return:
		"""
		x = int(round(np.cos(d * np.pi / 180) * r, 1))
		y = int(round(np.sin(d * np.pi / 180) * r, 1))
		return x, y
	
	@staticmethod
	def map_coords(points, position):
		return points[0] + position[0], points[1] + position[1]
	
	def prepare_data(self, data, position):
		xy_data = []
		for q, d, r in data:
			x, y = self.get_coords(d, r)
			x, y = self.map_coords((x, y), position)
			xy_data.append((x, y))
		return np.array(xy_data)
	
	@staticmethod
	def draw_main_map(map, data, position, color=200, thickness=2):
		for x, y in data:
			line(map, position, (x, y), color, thickness)
		return map
	
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
