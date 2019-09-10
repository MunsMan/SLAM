import cv2
import numpy as np
from modules import Lidar, LidarFunctions, Rotation, Motion, ImageQueue, Filter
from multiprocessing import Process, Queue, Event
import time
import pandas as pd
import sys

lidar_data = Queue()

ImgQueue = ImageQueue(10, 2)

lidar = Lidar(lidar_data)
lf = LidarFunctions()
print(lidar.start_lidar())
size = (5000, 5000)
position = (size[0] // 2, size[1] // 2)

new_scan_event = Event()
get_lidar_data = Process(target=lidar.get_scan_v3, args=(new_scan_event, 6))
start_time = time.time()
last_line_map = None
last_data_set = None
last_rotation = None
rotation_counter = 0
new_scan_event.set()
last_point_cloud = np.zeros(size, np.uint8)

try:
	get_lidar_data.start()
	
	while True:
		if not lidar_data.empty():
			print("Scan-time:", time.time() - start_time)
			start_time = time.time()
			i, data = lidar_data.get()
			pre_data = lf.prepare_data(data, position)
			mainMap = lf.draw_main_map(pre_data, position, size, np.zeros(size, np.uint8), 0)
			ImgQueue.put(mainMap)
			Image = ImgQueue.get_image()
			print(pre_data.shape)
			pointcloud = lf.draw_point_cloud(pre_data, size)
			raw_mask = np.array(Motion(last_point_cloud, pointcloud).motion_detection(100, 0), dtype=np.uint8)
			bmask = cv2.resize(raw_mask, (5000, 5000))
			assert pointcloud.shape == bmask.shape
			mask_img = np.copy(pointcloud)
			np.putmask(mask_img, bmask == False, 0)
			mask = cv2.resize(cv2.threshold(raw_mask, 0, 255, cv2.THRESH_BINARY)[1], (5000, 5000))
			print("Mask:", mask.shape)
			Image = np.vstack((np.hstack((pointcloud, last_point_cloud)), np.hstack((mask, Image))))
			new_scan_event.set()
			cv2.imshow("Masked Image", cv2.resize(np.array(pointcloud, dtype=np.uint8), (500, 500)))
			cv2.imshow("Karte", cv2.resize(Image, (1000, 1000)))
			key = cv2.waitKey(1)
			if key == ord('q'):
				print(pre_data.shape)
				print(last_data_set.shape)
				data = np.hstack((pre_data, last_data_set))
				pd.DataFrame(data).to_csv("data_set.csv")
				data = pd.read_csv("data_set.csv")
				print(data.shape)
				cv2.imwrite("Image.jpg", Image)
				print("END")
				cv2.destroyAllWindows()
				break
			last_data_set = pre_data
			last_point_cloud = pointcloud

finally:
	lidar.stop()
	get_lidar_data.terminate()
	print("exit")
	exit()
