import cv2
import numpy as np
from modules import Lidar, LidarFunktions, Feature
from multiprocessing import Process, Queue


def process_maps(mainMap, lineMap, lidar_data, positionMap):
	print("data")
	lmap = np.zeros((10000, 10000), np.uint8)
	mmap = mainMap.get()
	i, data = lidar_data.get()
	position = positionMap.get()
	pre_data = lf.prepare_data(data, position)
	mmap = lf.draw_main_map(mmap, pre_data, position)
	lmap = lf.draw_line_map(lmap, pre_data)
	mainMap.put(mmap)
	lineMap.put(lmap)
	print(lineMap.empty())
	exit()


def analyse_map(lineMap, plineMap_Queue):
	print("analyse")
	feat = Feature(lineMap.get(), 5)
	plmap, points = feat.get_corners()
	plineMap_Queue.put(plmap)
	exit()


lidar_data = Queue()
mainMap = Queue()
lineMap = Queue()
plineMap_Queue = Queue()
positionMap = Queue()

lidar = Lidar(lidar_data)
get_lidar_data = Process(target=lidar.get_scan)

lf = LidarFunktions()

mainMap.put(np.zeros((10000, 10000), np.uint8))
positionMap.put((5000, 5000))

print(lidar.start_lidar())

try:
	get_lidar_data.start()
	while True:
		if not lidar_data.empty():
			Process(target=process_maps, args=(mainMap, lineMap, lidar_data, positionMap)).start()
		if not lineMap.empty():
			Process(target=analyse_map, args=(lineMap, plineMap_Queue)).start()
		else:
			print("empty")
		if not plineMap_Queue.empty():
			image = np.hstack((mainMap.get(), plineMap_Queue.get()))
			cv2.imshow("Karte", cv2.resize(image, (1000, 500)))
			key = cv2.waitKey(1)
			if key == ord('q'):
				print("END")
				cv2.destroyAllWindows()
				break
			if key == ord('s'):
				cv2.imwrite("img.jpg", image)


finally:
	lidar.stop()
	print("exit")
	exit()
