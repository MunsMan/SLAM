import cv2
import numpy as np
from modules import Lidar, LidarFunktions
from multiprocessing import Process, Queue
import time

# Queue
lidar_data = Queue()

lidar = Lidar(lidar_data)
lf = LidarFunktions()
print(lidar.start_lidar())
size = (5000, 5000)
position = (size[0] // 2, size[1] // 2)
mainMap = np.zeros(size, np.uint8)

get_lidar_data = Process(target=lidar.get_scan)

try:
	get_lidar_data.start()
	
	while True:
		if not lidar_data.empty():
			st = time.time()
			lineMap = np.zeros(size, np.uint8)
			print("creating map:", time.time() - st)
			i, data = lidar_data.get()
			pre_data = lf.prepare_data(data, position)
			print("Pre_data:", pre_data.shape)
			mainMap = lf.draw_main_map_static(mainMap, pre_data, position, size)
			lineMap = lf.draw_line_map(lineMap, pre_data)
			image = np.hstack((mainMap, lineMap))
			cv2.imshow("Karte", cv2.resize(image, (1000, 500)))
			key = cv2.waitKey(1)
			if key == ord('q'):
				print("END")
				cv2.destroyAllWindows()
				break
			if key == ord('s'):
				cv2.imwrite("img{i}.jpg".format(i=i), lineMap)
			print(time.time() - st)
		time.sleep(0.05)

finally:
	lidar.stop()
	print("exit")
	exit()
