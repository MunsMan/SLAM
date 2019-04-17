from modules import Lidar, LidarFunktions
import multiprocessing as mp
import numpy as np
import ctypes as c
import cv2


class GenerateMaps:
	
	def __init__(self, size, image_array, position):
		self._buffer = mp.Queue()
		self.lidar = Lidar(self._buffer)
		print(self.lidar.start_lidar())
		self.lf = LidarFunktions()
		self.size = size
		self.position = position
		self.shared_line_map_array = image_array
	
	def get_line_map(self, color=200, thickness=5):
		lidar_data = mp.Process(target=self.lidar.get_scan)
		lidar_data.start()
		while not self._buffer.empty():
			i, data = self._buffer.get()
			pre_data = self.lf.prepare_data(data, position)
			line_map = self.lf.draw_line_map(np.zeros(self.size, dtype=np.uint8), pre_data, color, thickness)
			self.shared_line_map_array[:] = line_map.reshape(-1)
			print(self.shared_line_map_array[:])


if __name__ == '__main__':
	size = (5000, 5000)
	arr = mp.Array(c.c_uint8, size[0] * size[1])
	position = mp.Array(c.c_uint16, 2)
	position[:] = (size[0] // 2, size[1] // 2)
	gm = GenerateMaps(size, arr, position)
	last_image = np.zeros(size, dtype=np.uint8)
	get_line_map = mp.Process(target=gm.get_line_map)
	get_line_map.start()
	while True:
		image = np.frombuffer(arr.get_obj(), np.uint8).reshape(size)
		if np.mean(last_image == image) == 1:
			continue
		else:
			image = cv2.resize(image, (400, 400))
			cv2.imshow("Karte", image)
			key = cv2.waitKey(1)
			if key == ord("q"):
				cv2.destroyAllWindows()
				break
			last_image = image
	gm.lidar.stop()
	for process in mp.active_children():
		process.terminate()
	exit()
