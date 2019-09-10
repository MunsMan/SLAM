import cv2
import numpy as np
from multiprocessing import Process, Queue, Event
import time
from api.Karte import Map
from api.LidarControll import LidarControll

Lidar = LidarControll(size=3000)
Map = Map()

Lidar.start()
Lidar.set_event()
first_set = Lidar.get_data()
Lidar.set_event()
try:
	while True:
		second_set = Lidar.get_data()
		second_set_t = Map.transform_set(second_set,
		                                 Map.icp(first_set,
		                                         second_set,
		                                         amount_of_points=100,
		                                         max_iter=15,
		                                         eps=100,
		                                         show_animation=True))
		main_map, point_cloud_arr = Lidar.create_maps(second_set_t)
		Lidar.set_event()
		first_set = second_set
		if Lidar.show_images(np.hstack((main_map, point_cloud_arr)), (1200, 600)):
			break
finally:
	Lidar.stop()
