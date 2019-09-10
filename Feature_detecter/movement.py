import cv2
import numpy as np
from modules import Lidar, LidarFunctions, Rotation, Motion, ImageQueue
from multiprocessing import Process, Queue, Event
import time
from imutils import rotate

lidar_data = Queue()

ImgQueue = ImageQueue(10, 1)

lidar = Lidar(lidar_data)
lf = LidarFunctions()
print(lidar.start_lidar())
size = (5000, 5000)
position = (size[0] // 2, size[1] // 2)
mainMap = np.zeros(size, np.uint8)

new_scan_event = Event()
get_lidar_data = Process(target=lidar.get_scan_v3, args=(new_scan_event, 6))
start_time = time.time()
last_line_map = None
last_image = None
last_rotation = None
rotation_counter = 0
new_scan_event.set()

Motion = Motion(size)

try:
	get_lidar_data.start()
	
	while True:
		if not lidar_data.empty():
			print("Scan-time:", time.time() - start_time)
			start_time = time.time()
			i, data = lidar_data.get()
			pre_data = lf.prepare_data(data, position)

finally:
	lidar.stop()
	get_lidar_data.terminate()
	print("exit")
	exit()
