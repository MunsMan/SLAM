import numpy as np
import cv2
import time
from modules import Lidar
import multiprocessing as mp
from Process import process_lidar_data
import ctypes as c

# Variablen
size = (5000, 5000)

# Queue
lidar_data_queue = mp.Queue()

# Shard Memory
position = mp.Array(c.c_uint16, 2)
rotation = mp.Value(c.c_float)
pre_data_shared = mp.Array(c.c_uint16, 362 * 2)
lineMap_shared = mp.Array(c.c_uint8, size[0] * size[1] + 1)
mainMap_array = mp.Array(c.c_uint8, size[0] * size[1])

position[:] = [size[0] // 2, size[1] // 2]
rotation.value = 0

# Classes and Funktion
lidar = Lidar(lidar_data_queue)

# Processes
get_lidar_data = mp.Process(target=lidar.get_scan)
process_data = mp.Process(target=process_lidar_data,
                          args=(lidar_data_queue,
                                pre_data_shared,
                                lineMap_shared,
                                mainMap_array,
                                rotation,
                                size))

print(lidar.start_lidar())
get_lidar_data.start()
process_data.start()

try:
	time.sleep(1)
	last_mainMap = np.zeros(size, dtype=np.uint8)
	st = time.time()
	while True:
		mainMap = np.frombuffer(mainMap_array.get_obj(), np.uint8).reshape(size)
		if np.mean(last_mainMap == mainMap) == 1:
			continue
		else:
			mainMap = cv2.resize(mainMap, (400, 400))
			cv2.imshow("Karte", mainMap)
			key = cv2.waitKey(1)
			if key == ord('q'):
				print("END")
				cv2.destroyAllWindows()
				break
			last_mainMap = mainMap
			print("Running Processes:", len(mp.active_children()))
		time.sleep(0.1)
	
finally:
	lidar.stop()
	print("exit")
	for p in mp.active_children():
		p.terminate()
	exit()
