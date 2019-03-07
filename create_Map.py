import cv2
import numpy as np
from multiprocessing import Process, Queue
from modules import Lidar
import time


def main_process():
	buffer_empty_counter = 0
	while True:
		if not buffer.empty():
			buffer_empty_counter = 0
			i, scan = buffer.get()
			for _, grad, distance in scan:
				points = lidar.get_coords(grad, distance)
				coords = lidar.map_coords(points, position)
				cv2.line(mainMap, position, coords, 135)
			if i % 3 == 1:
				resized = cv2.resize(mainMap, (400, 400))
				cv2.imshow("Karte", resized)
			key = cv2.waitKey(1)
			if key == ord('q'):
				cv2.destroyAllWindows()
				break
			elif key == ord('s'):
				cv2.imwrite("img{i}.jpg".format(i=i), mainMap)

		else:
			print("Buffer empty")
			time.sleep(0.1)
			buffer_empty_counter += 1
			if buffer_empty_counter > 8:
				exit()


if __name__ == '__main__':

	mainMap = np.zeros((10000, 10000))
	position = (5000, 5000)
	buffer = Queue()
	lidar = Lidar(buffer)
	print(lidar.start_lidar())
	lidar_scan = Process(target=lidar.get_scan)
	main_p = Process(target=main_process)

	try:
		print("start")
		main_p.start()
		lidar_scan.start()
		while True:
			if not main_p.is_alive():
				lidar_scan.terminate()
				lidar.stop()
				exit()
			time.sleep(1)

	except KeyboardInterrupt:
		main_p.kill()
		lidar_scan.kill()
		lidar.stop()
		exit()
