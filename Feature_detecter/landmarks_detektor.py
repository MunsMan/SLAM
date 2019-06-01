import cv2
import numpy as np
from modules import Lidar, LidarFunktions
from multiprocessing import Process, Queue, Event, Array
import time
import math


def draw_map(data, size):
	image = np.zeros(size, dtype=np.uint8)
	for x, y in data:
		cv2.line(image, (x, y), (x, y), 255, 10)
	return image


def gerade_berechnen(x1, y1, x2, y2):
	m = (y1 - y2) / (x1 - x2)
	t = y1 - m * x1
	return m, t


def find_lotpunkt(x1, y1, x2, y2, p1, p2):
	r = (-((x1 - p1) * (x1 - x2) + (y1 - p2) * (y1 - y2))) / ((x1 - x2) ** 2 + (y1 - y2) ** 2)
	l1 = int(x1 + r * (x1 - x2))
	l2 = int(y1 + r * (y1 - y2))
	return l1, l2


def get_degree(l1, l2, p1, p2):
	if l2 < p2:
		grad = math.atan((p2 - l2) / (p1 - l1)) / (2 * math.pi) * 360
		if grad > 0:
			return grad
		else:
			return 180 + grad
	else:
		grad = math.atan((p2 - l2) / (p1 - l1)) / (2 * math.pi) * 360
		if grad > 0:
			return grad + 180
		else:
			return 360 + grad


def mark_line(image, position):
	flag, b = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
	edges = cv2.Canny(b, 150, 200, 3, 5)
	lines = cv2.HoughLinesP(edges, 10, np.pi / 128, 290, minLineLength=400, maxLineGap=100)
	if lines is not None:
		lines = lines.reshape((-1, 4))
		color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		len_max = 0
		for x1, y1, x2, y2 in lines:
			lenght = math.sqrt(abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2)
			if len_max < lenght:
				len_max = lenght
				line = (x1, y1, x2, y2)
		cv2.line(color_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 10)
		l1, l2 = find_lotpunkt(line[0], line[1], line[2], line[3], position[0], position[1])
		cv2.line(color_image, (l1, l2), position, (255, 0, 0), 10)
		cv2.line(color_image, (0, 2500), (2500, 2500), (0, 0, 255), 4)
		cv2.line(color_image, (5000, 2500), (2500, 2500), (0, 255, 0), 4)
		grad = get_degree(l1, l2, position[0], position[1])
	else:
		return image, None
	return color_image, grad


# Queue
lidar_data = Queue()

lidar = Lidar(lidar_data)
lf = LidarFunktions()
print(lidar.start_lidar())
size = (5000, 5000)
position = (size[0] // 2, size[1] // 2)
mainMap = np.zeros(size, np.uint8)

# mainMapArray = Array()
mainMapEvent = Event()
new_scan_event = Event()
get_lidar_data = Process(target=lidar.get_scan_v3, args=(new_scan_event, 2))
start_time = time.time()
last_line_map = None
grad_counter = 0
last_grad = None

try:
	get_lidar_data.start()
	
	while True:
		if not lidar_data.empty():
			print("Scan-time:", time.time() - start_time)
			i, data = lidar_data.get()
			print(len(data))
			pre_data = lf.prepare_data(data, position)
			mainMap = lf.draw_main_map_static(mainMap, pre_data, position, grad_counter, size, i)
			image = draw_map(pre_data, size)
			image, grad = mark_line(image, position)
			if grad is not None:
				if last_grad is not None:
					dgrad = grad - last_grad
					grad_counter += dgrad
					last_grad = grad
				else:
					last_grad = grad
			
			new_scan_event.set()
			
			image = np.hstack((image, cv2.cvtColor(mainMap, cv2.COLOR_GRAY2BGR)))
			cv2.imshow("Karte", cv2.flip(cv2.resize(image, (2000, 1000)), 1))
			key = cv2.waitKey(1)
			if key == ord('q'):
				print("END")
				cv2.destroyAllWindows()
				break
			if key == ord('s'):
				cv2.imwrite("image{i}.jpg".format(i=i), image)
			start_time = time.time()
			new_scan_event.clear()

finally:
	lidar.stop()
	get_lidar_data.terminate()
	print("exit")
	exit()
