"""
Program that only sends data 5 time a second. So we can adjust the time to the Computetime of the PC,
to get mostly lag free images. So the Information we extract from the images are still relevant.
"""
from rplidar import RPLidar as rp
import time
import multiprocessing
import cv2
import numpy as np


class Lidar:
	
	def __init__(self, buffer):
		self.__buffer__ = buffer
		self.lidar = rp('/dev/tty.SLAB_USBtoUART')
		time.sleep(2)
	
	def scan(self):
		inter = self.lidar.iter_measurments(0)
		scan = []
		st = time.time()
		try:
			for i in inter:
				n, q, a, d = i
				if n:
					if time.time() > st:
						self.__buffer__.put(scan)
						print("Scan")
						st += 0.2
					scan = []
					scan.append((q, a, d))
				else:
					scan.append((q, a, d))
		finally:
			self.stop()
	
	def stop(self):
		self.lidar.stop()
		self.lidar.stop_motor()
		self.lidar.disconnect()


class LidarFunktions:
	def __init__(self, position=(3500, 3500)):
		self.position = position
	
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
	def draw_line_map(map, data, color=200, thickness=5):
		last_point = None
		for x, y in data:
			if last_point is None:
				last_point = (x, y)
			else:
				cv2.line(map, last_point, (x, y), color, thickness)
				last_point = (x, y)
		return map
	
	def main(self, buffer):
		while True:
			if not buffer.empty():
				print("Scan1")
				data = buffer.get()
				pre_data = self.prepare_data(data=data, position=self.position)
				zeroMap = np.zeros((7000, 7000), np.uint8)
				image = self.draw_line_map(zeroMap, pre_data)
				cv2.imshow("Image", cv2.resize(image, (400, 400)))
				key = cv2.waitKey(1)
				if key == ord('q'):
					cv2.destroyAllWindows()
					self.stop()
			else:
				time.sleep(0.1)
	
	def stop(self):
		for i in multiprocessing.active_children():
			i.terminate()
			print(i)
		exit()


buffer = multiprocessing.Queue()

p1 = multiprocessing.Process(target=Lidar(buffer).scan)
p1.start()
LidarFunktions().main(buffer)
