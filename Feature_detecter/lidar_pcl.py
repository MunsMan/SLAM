from multiprocessing import Process, Queue, Event
import numpy as np
import pcl
import time
from rplidar import RPLidar as rp


class Lidar:
	
	def __init__(self, buffer):
		self.lidar = rp("/dev/tty.SLAB_USBtoUART")
		self.__buffer = buffer
	
	def start_lidar(self):
		self.lidar.start_motor()
		info = self.lidar.get_info()
		time.sleep(2)
		return info
	
	def get_scan_v3(self, event, rotations=1):
		inter = self.lidar.iter_measurments(0)
		scan = []
		scan_counter = 1
		rotation = 0
		try:
			for i in inter:
				n, q, a, d = i
				if n:
					if event.is_set() and rotation >= rotations:
						scan.append((q, a, d))
						self.__buffer.put((scan_counter, scan))
						event.clear()
						rotation = 0
						scan = []
					scan_counter += 1
					rotation += 1
				else:
					scan.append((q, a, d))
		finally:
			self.stop()
	
	def stop(self):
		self.lidar.stop()
		self.lidar.stop_motor()
		self.lidar.disconnect()
	
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
	
	def prepare_data_to_3D(self, data, position):
		xy_data = []
		for q, d, r in data:
			x, y = self.get_coords(d, r)
			x, y = self.map_coords((x, y), position)
			xy_data.append((x, y, 0))
		return np.array(xy_data)


lidar_data = Queue()

lidar = Lidar(lidar_data)

print(lidar.start_lidar())
size = (5000, 5000)
position = (size[0] // 2, size[1] // 2)
mainMap = np.zeros(size, np.uint8())

new_scan_event = Event()
get_lidar_data = Process(target=lidar.get_scan_v3, args=(new_scan_event, 3))
new_scan_event.set()
last_point_array = None

try:
	get_lidar_data.start()
	
	i, data = lidar_data.get()
	pre_data = lidar.prepare_data_to_3D(data, position)
	last_point_array = pcl.PointCloud(np.array(pre_data, np.float32))
	new_scan_event.set()
	
	while True:
		if not lidar_data.empty():
			i, data = lidar_data.get()
			pre_data = lidar.prepare_data_to_3D(data, position)
			p = pcl.PointCloud(np.array(pre_data, np.float32))
			
			new_scan_event.set()
finally:
	lidar.stop()
	get_lidar_data.terminate()
	print("exit")
	exit()
