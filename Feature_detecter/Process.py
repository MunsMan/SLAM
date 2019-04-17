from modules import LidarFunktions
from modules import Feature, Filter, Matches, Motion
import cv2
from imutils import rotate
import numpy as np
import multiprocessing as mp
import time
import ctypes as c


def process_lidar_data(lidar_data_queue, pre_data_array, lineMap_array, mainMap_array, rotation, size):
	position = (size[0] // 2, size[1] // 2)
	while True:
		if not lidar_data_queue.empty():
			st = time.time()
			data_id, data = lidar_data_queue.get()
			lf = LidarFunktions()
			pre_data = lf.prepare_data(data, position)
			pre_data_array[0] = data_id
			pre_data_array[1] = pre_data.size
			pre_data_array[2:pre_data.size + 2] = pre_data.reshape(-1)
			mp.Process(target=process_data,
			           args=(size,
			                 pre_data_array,
			                 lineMap_array,
			                 mainMap_array,
			                 rotation,
			                 position)).start()


def line_map(size, pre_data_array, lineMap_array):
	lf = LidarFunktions()
	data_id = pre_data_array[0]
	data_size = int(pre_data_array[1])
	data = np.frombuffer(pre_data_array.get_obj(), c.c_uint16)[2:(data_size + 2)].reshape(-1, 2)
	lineMap = lf.draw_line_map(np.zeros(size, dtype=c.c_uint8), data)
	lineMap_array[0] = data_id
	lineMap_array[1:] = lineMap.reshape(-1)
	print("LineMap:", data_id)


def main_map(size, pre_data_array, rotation, position, mainMap_array):
	lf = LidarFunktions()
	data_id = pre_data_array[0]
	data_size = pre_data_array[1]
	position_data = (position[0], position[1])
	grad = rotation
	mainMap = np.frombuffer(mainMap_array.get_obj(), c.c_uint8).reshape(size)
	mainMap = lf.draw_main_map(np.frombuffer(pre_data_array.get_obj(), c.c_uint16)[2:].reshape(-1, 2),
	                           position_data,
	                           size,
	                           mainMap,
	                           grad)
	mainMap_array[:] = mainMap.reshape(-1)


# print("MainMap:", data_id)


def process_data(size, pre_data_array, lineMap_array, mainMap_array, rotation, position):
	last_id = lineMap_array[0]
	last_lineMap = np.frombuffer(lineMap_array.get_obj(), c.c_uint8)[1:].reshape(size)
	lineMap_process = mp.Process(target=line_map, args=(size, pre_data_array, lineMap_array))
	mainMap_process = mp.Process(target=main_map, args=(size, pre_data_array, rotation.value, position, mainMap_array))
	lineMap_process.start()
	lineMap_process.join()
	lineMap = np.frombuffer(lineMap_array.get_obj(), c.c_uint8)[1:].reshape(size)
	num_matches, grad, _ = cal_rot_move(last_lineMap, lineMap, 0)
	rotation.value = rotation.value + grad
	print("Matches:", num_matches, "Drehung:", grad)
	mainMap_process.start()


def cal_rot_move(img1, img2, thresh):
	feat = Feature(img1, img2, 10)
	
	good, points1, points2, kp1, kp2 = feat.main()
	
	points1, points2 = Filter(points1, points2, thresh).filter_points()
	
	num_matches = points1.shape[0]
	
	print("Matches:", num_matches)
	
	matches = Matches(feat.img1, points1, feat.img2, points2)
	
	motion = Motion(points1, points2)
	
	dgrad, line1, line2 = motion.rotation()
	
	cv2.line(feat.img1, line1[0], line1[1], (0, 0, 255))
	cv2.line(feat.img2, line2[0], line2[1], (0, 255, 0))
	
	print("Grad", dgrad[0, 0])
	
	feat.img2 = rotate(feat.img2, dgrad, (250, 250))
	
	img3 = matches.match_drawer()
	img4 = cv2.add(feat.img1, feat.img2)
	
	_img5 = np.hstack((img3, img4))
	
	return num_matches, dgrad[0, 0], _img5
