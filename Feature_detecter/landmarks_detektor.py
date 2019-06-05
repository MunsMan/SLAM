import cv2
import numpy as np
from modules import Lidar, LidarFunctions, Rotation
from multiprocessing import Process, Queue, Event, Array
import time

# Queue
lidar_data = Queue()

lidar = Lidar(lidar_data)
lf = LidarFunctions()
print(lidar.start_lidar())
size = (5000, 5000)
position = (size[0] // 2, size[1] // 2)
mainMap = np.zeros(size, np.uint8)

# mainMapArray = Array()
mainMapEvent = Event()
new_scan_event = Event()
get_lidar_data = Process(target=lidar.get_scan_v3, args=(new_scan_event, 3))
start_time = time.time()
last_line_map = None
rotation_counter = 0
last_rotation = None
new_scan_event.set()

try:
	get_lidar_data.start()
	
	while True:
		if not lidar_data.empty():
			print("Scan-time:", time.time() - start_time)
			start_time = time.time()
			i, data = lidar_data.get()
			pre_data = lf.prepare_data(data, position)
			mainMap = lf.draw_main_map_static(mainMap, pre_data, position, rotation_counter, size, 0.80, i)
			rotation_return = Rotation(position, size).main(pre_data, True, last_rotation, rotation_counter)
			image, last_rotation, rotation_counter = rotation_return
			new_scan_event.set()
			image = np.hstack((image, cv2.cvtColor(mainMap, cv2.COLOR_GRAY2BGR)))
			cv2.imshow("Karte", cv2.resize(image, (1000, 500)))
			key = cv2.waitKey(1)
			if key == ord('q'):
				print("END")
				cv2.destroyAllWindows()
				break
			if key == ord('s'):
				cv2.imwrite("image{i}.jpg".format(i=i), image)

finally:
	lidar.stop()
	get_lidar_data.terminate()
	print("exit")
	exit()
