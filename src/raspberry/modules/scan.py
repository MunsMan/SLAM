from rplidar import RPLidar as rp
import time
import numpy as np


class Lidar:
	
	def __init__(self, buffer):
		"""
		Function to controll the Lidar Sensor. Writen especially for the RPLidar A2M8
		:param buffer: Queue() - used for multiprocessing
		"""
		self.lidar = rp("/dev/ttyUSB0")
		# self.lidar = rp("/dev/tty.SLAB_USBtoUART")
		self.__buffer = buffer
	
	def start_lidar(self):
		"""
		Function to start the Lidars Motor and gather info.
		This function is giving the Lidar the necessary to start everything.
		:return: string() -  the gathered info
		"""
		self.lidar.start_motor()
		info = self.lidar.get_info()
		time.sleep(2)
		return info
	
	def get_scan(self):
		
		inter = self.lidar.iter_measurments(0)
		scan = []
		st = time.time()
		scan_counter = 1
		try:
			for i in inter:
				n, q, a, d = i
				if len(scan) == 360:
					self.__buffer.put((scan_counter, np.array(scan, dtype=np.int64)))
					scan_counter += 1
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
