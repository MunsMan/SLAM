import cv2
import numpy as np
from modules import Lidar
from multiprocessing import Process, Queue


def use_data():
	while True:
		if not queue.empty():
			i, data = queue.get()
			# q: quality d: degree, r: radius
			for q, d, r in data:
				x, y = lidar.get_coords(d, r)


queue = Queue()
lidar = Lidar(queue)
lidar.start_lidar()

get_lidar_data = Process(target=lidar.get_scan)
use_lidar_data = Process(target=use_data)

try:
	get_lidar_data.start()
	use_lidar_data.start()
finally:
	lidar.stop()
