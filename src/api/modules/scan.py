import numpy as np
from rplidar import RPLidar as rp
import time


class Lidar:
	
	def __init__(self, buffer):
		"""
		Function to controll the Lidar Sensor. Writen especially for the RPLidar A2M8
		:param buffer: Queue() - used for multiprocessing
		"""
		self.lidar = rp("/dev/tty.SLAB_USBtoUART")
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
		"""
		First prototype for the LIDAR function. Just iterating over the enumerated data and pushing it into the buffer.
		:return: tuple(int(), list()) - Counter of Scans, list of result(quality, degree, distance)
		"""
		try:
			for i, scan in enumerate(self.lidar.iter_scans()):
				self.__buffer.put((i, scan))
				time.sleep(0.2)
				print("Scaned:", i)
		
		finally:
			self.stop()
	
	def get_scan_v2(self, sps):
		"""
		Function receive the data from the Lidar in a defined frequency. You define it in scans per Second.
		:param sps: int() - scans per seconds
		:return:
		"""
		inter = self.lidar.iter_measurments(0)
		scan = []
		st = time.time()
		tps = 1 / sps
		scan_counter = 1
		try:
			for i in inter:
				n, q, a, d = i
				if n:
					if time.time() > st:
						self.__buffer.put((scan_counter, scan))
						st += tps
						scan_counter += 1
					scan = []
					scan.append((q, a, d))
				else:
					scan.append((q, a, d))
		finally:
			self.stop()
	
	def get_scan_v3(self, event, rotations=1):
		"""
		Function to start the scanning Process. This function will scan for the given amount of Rotations.
		It will push the collected data on to the buffer, when the event is set.
		:param event: event() - push the data if set
		:param rotations: int() - the amount of rotations which should be pushed
		:return: None - the data is pushed to the buffer
		"""
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
					elif rotation >= rotations:
						rotation = 0
						scan = []
						print("Del")
					scan_counter += 1
					rotation += 1
				else:
					scan.append((q, a, d))
		finally:
			self.stop()
	
	def stop(self):
		"""
		This function is used to stop and close the Lidar correctly.
		:return:
		"""
		self.lidar.stop()
		self.lidar.stop_motor()
		self.lidar.disconnect()
