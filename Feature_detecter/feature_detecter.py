import cv2
import numpy as np
from modules import Lidar, LidarFunctions
from multiprocessing import Process, Queue, Event
import time
from Process import cal_rot_move


# Queue
lidar_data = Queue()

lidar = Lidar(lidar_data)
lf = LidarFunktions()
print(lidar.start_lidar())
size = (5000, 5000)
position = (size[0] // 2, size[1] // 2)
mainMap = np.zeros(size, np.uint8)

new_scan_event = Event()
get_lidar_data = Process(target=lidar.get_scan_v3, args=(new_scan_event,))
start_time = time.time()
last_line_map = None
grad_counter = 0


try:
	get_lidar_data.start()
	
	while True:
		if not lidar_data.empty():
			st = time.time()
			i, data = lidar_data.get()
			pre_data = lf.prepare_data(data, position)
			mainMap = lf.draw_main_map_static(mainMap, pre_data, position, grad_counter, size)
			lineMap = lf.draw_line_map(np.zeros(size, np.uint8), pre_data)
			if last_line_map is not None:
				num_matches, grad, img2 = cal_rot_move(last_line_map, lineMap, 50)
				new_scan_event.set()
				print("Matches:", num_matches, "Grad", grad)
				if num_matches < 20 or grad > 5:
					continue
				else:
					grad_counter += grad
				new_scan_event.clear()
			else:
				img2 = np.zeros((5000, 15000))
			image = cv2.resize(np.hstack((mainMap, lineMap, np.zeros(size))), (1500, 500))
			image = np.vstack((image, cv2.resize(img2, (1500, 500))))
			cv2.imshow("Karte", image)
			key = cv2.waitKey(1)
			if key == ord('q'):
				print("END")
				cv2.destroyAllWindows()
				break
			if key == ord('s'):
				cv2.imwrite("lineMap{i}.jpg".format(i=i), lineMap)
			last_line_map = lineMap
			print("Grad Counter:", grad_counter)
		# print(time.time() - st)
		# print("Finished:", i)

finally:
	lidar.stop()
	get_lidar_data.terminate()
	print("exit")
	exit()
