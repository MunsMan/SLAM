import cv2
import numpy as np
from modules import Lidar, LidarFunctions, Rotation, Motion
from multiprocessing import Process, Queue, Event
import time
from imutils import rotate


def draw_map(data, size):
	image = np.zeros(size, dtype=np.uint8)
	for x, y in data:
		cv2.line(image, (x, y), (x, y), 255, 10)
	return image


def movement(last_pre_data, pre_data, rotation, size):
	last_cp = draw_map(last_pre_data, size)
	cp = draw_map(pre_data, size)
	cp = rotate(cp, rotation)
	
	


# Queue
lidar_data = Queue()

lidar = Lidar(lidar_data)
lf = LidarFunctions()
print(lidar.start_lidar())
size = (5000, 5000)
position = (size[0] // 2, size[1] // 2)
mainMap = np.zeros(size, np.uint8)

# mainMapArray = Array()
# mainMapEvent = Event()
new_scan_event = Event()
get_lidar_data = Process(target=lidar.get_scan_v3, args=(new_scan_event, 3))
start_time = time.time()
last_line_map = None
rotation_counter = 0
last_rotation = None
new_scan_event.set()
last_image = None

try:
	get_lidar_data.start()
	
	while True:
		if not lidar_data.empty():
			print("Scan-time:", time.time() - start_time)
			start_time = time.time()
			i, data = lidar_data.get()
			pre_data = lf.prepare_data(data, position)
			print("Datasize:", pre_data.shape)
			rotation_return = Rotation(position, size).main(pre_data, True, last_rotation, rotation_counter, True)
			c_image, last_rotation, rotation_counter, image = rotation_return
			if last_image is not None:
				print(Motion(last_image, image, 10).movement2())
			mainMap = lf.draw_and_add_main_map(mainMap, pre_data, position, rotation_counter, size, 0.90, i)
			new_scan_event.set()
			c_image = np.hstack((c_image, cv2.cvtColor(mainMap, cv2.COLOR_GRAY2BGR)))
			last_image = image
			cv2.imshow("Karte", cv2.resize(c_image, (1000, 500)))
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
