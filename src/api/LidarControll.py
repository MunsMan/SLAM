import multiprocessing as mp
from .modules import Lidar, lf
import numpy as np
import cv2


class LidarControll:
	
	def __init__(self, size: int() = 3000) -> None:
		self.queue = mp.Queue()
		self.lidar = Lidar(self.queue)
		self.lf = lf()
		self.new_scan = mp.Event()
		self.size = (size, size)
		self.position = (size // 2, size // 2)
		self.process = self._get_lidar_data()
		self.main_arr = mp.Array("I", self.size[0] ** 2)
		self.pointcloud_arr = mp.Array("I", self.size[0] ** 2)
		self.last_pre_data = None
		self.last_pointcloud = np.zeros(self.size, np.uint8)
		self.pre_data = None
		self.main_map = np.zeros(self.size, np.uint8)
	
	def start(self):
		print(self.lidar.start_lidar())
		process = self._get_lidar_data()
		self._get_lidar_data_start()
	
	def stop(self):
		self._get_lidar_data_stop()
		self.lidar.stop()
	
	def _get_lidar_data(self):
		return mp.Process(target=self.lidar.get_scan_v3, args=(self.new_scan, 6))
	
	def _get_lidar_data_start(self):
		self.process.start()
	
	def _get_lidar_data_stop(self):
		self.process.terminate()
	
	def set_event(self):
		self.new_scan.set()
	
	def _create_main_map(self, pre_data, array, position, size, main_map):
		mainMap = self.lf.draw_main_map(pre_data, position, size, main_map)
		array[:] = np.reshape(np.array(mainMap, np.uint32), -1)
	
	def _create_pointcloud(self, pre_data, array):
		pointcloud = self.lf.draw_point_cloud(pre_data, self.size)
		array[:] = np.reshape(np.array(pointcloud, np.uint32), -1)
	
	def get_data(self):
		i, data = self.queue.get()
		pre_data = self.lf.prepare_data(data, (0, 0))
		return pre_data
	
	def create_maps(self, pre_data):
		print("starting processes")
		p_main_map = mp.Process(target=self._create_main_map, args=(pre_data + self.position,
		                                                            self.main_arr,
		                                                            self.position,
		                                                            self.size,
		                                                            self.main_map))
		p_pointcloud = mp.Process(target=self._create_pointcloud, args=(pre_data + self.position,
		                                                                self.pointcloud_arr))
		p_main_map.start()
		p_pointcloud.start()
		p_main_map.join()
		p_pointcloud.join()
		main_map = np.array(np.frombuffer(self.main_arr.get_obj(), np.uint32).reshape(self.size), dtype=np.uint8)
		pointcloud_arr = np.array(np.frombuffer(self.pointcloud_arr.get_obj(), np.uint32).reshape(self.size),
		                          dtype=np.uint8)
		self.main_map = main_map
		return main_map, pointcloud_arr
	
	@staticmethod
	def show_images(image, image_size=(800, 400)):
		cv2.imshow("Image", cv2.resize(image, image_size))
		key = cv2.waitKey(5)
		if key == ord('q'):
			print("END")
			cv2.destroyAllWindows()
			return True
	
	def main(self):
		self.start()
		pointcloud_arr = None
		try:
			self._get_lidar_data_start()
			
			while True:
				if pointcloud_arr is not None:
					self.last_pointcloud = pointcloud_arr
				main_map, pointcloud_arr = self.create_maps()
				image = np.hstack((main_map, pointcloud_arr, self.last_pointcloud))
				self.set_event()
				self.last_pre_data = self.pre_data
				if self.show_images(image):
					break
		
		finally:
			self._get_lidar_data_stop()
			self.stop()


if __name__ == '__main__':
	LC = LidarControll()
	LC.main()
	print("Finished")
