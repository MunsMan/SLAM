from multiprocessing import Process, Queue, Event
from modules import Lidar, LidarFunctions, Rotation, ImageQueue
from imutils import rotate
import cv2
import numpy as np
import time


def draw_map(data, size):
	image = np.zeros(size, dtype=np.uint8)
	for x, y in data:
		cv2.line(image, (x, y), (x, y), 255, 10)
	return image


def rotate_point(points, rotation):
	ox, oy = (0, 0)
	px, py = points.T
	
	px = ox


def movement(last_pre_data, pre_data, rotation, size):
	last_cp = draw_map(last_pre_data, size)
	cp = draw_map(pre_data, size)
	cp = rotate(cp, rotation)


# Queue
lidar_data = Queue()

mainMap_img_q = ImageQueue(5, 2)
pci_img_q = ImageQueue(3, 2)

lidar = Lidar(lidar_data)
lf = LidarFunctions()
print(lidar.start_lidar())
size = (5000, 5000)
position = (size[0] // 2, size[1] // 2)
mainMap = np.zeros(size, np.uint8)

# mainMapArray = Array()
new_scan_event = Event()
get_lidar_data = Process(target=lidar.get_scan_v3, args=(new_scan_event, 3))
start_time = time.time()
last_line_map = None
rotation_counter = 0
last_rotation = 0
new_scan_event.set()

try:
	get_lidar_data.start()
	
	while True:
		if not lidar_data.empty():
			print("Scan-time:", time.time() - start_time)
			start_time = time.time()
			i, data = lidar_data.get()
			pre_data = lf.prepare_data(data, position)
			rotation_return = Rotation(position, size).main(pre_data, False, last_rotation, rotation_counter, True)
			image, last_rotation, rotation_counter, pci_img = rotation_return
			s, mainMap = lf.draw_main_map(mainMap, pre_data, position, rotation_counter, size, 0.80, i)
			if s is not None:
				mainMap = mainMap_img_q.place(mainMap, True)
			pci_img_q.place(pci_img, False)
			last_pci_img = pci_img_q.get_image()
			
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
