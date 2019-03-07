import numpy as np
from rplidar import RPLidar as rp
import time


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
	
	@staticmethod
	def get_coords(g, d):
		x = int(round(np.cos(g * np.pi / 180) * d, 1))
		y = int(round(np.sin(g * np.pi / 180) * d, 1))
		return x, y
	
	@staticmethod
	def map_coords(points, position):
		return points[0] + position[0], points[1] + position[1]
	
	def stop(self):
		self.lidar.stop()
		self.lidar.stop_motor()
		self.lidar.disconnect()
