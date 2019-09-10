import numpy as np
import cv2


# from .feature import Feature
# from .filter import Filter

class Motion:
	
	def __init__(self, base_img, new_img, scale=10):
		self.scale = scale
		self.base_img = base_img
		self.new_img = new_img
		self.weight = 0.00001
		self.thresh = 0.95
	
	def cal_error(self, new_img):
		b_base_img = np.isin(self.base_img, np.max(self.base_img))
		b_new_img = np.isin(new_img, np.max(new_img))
		derror1 = np.sum(np.bitwise_and(b_new_img, b_base_img)) / np.nonzero(b_new_img)[0].shape[0]
		new_img_center = self.center(b_new_img)
		# derror2 = math.sqrt((self.base_center[0] - new_img_center[0]) ** 2 + abs(self.base_center[1] - new_img_center[1]) ** 2)
		error = (derror1 + self.weight)  # + 1 / ((derror2 + self.weight)*10)
		return error
	
	@staticmethod
	def center(array):
		pts = np.nonzero(array)
		return np.array([np.mean(pts[0]), np.mean(pts[1])])
	
	def shift(self, array, direction, step=10):
		shifted = np.roll(array, direction[0] * step, axis=direction[1])
		if direction[1] == 0:
			if direction[0] < 0:
				shifted[shifted.shape[0] - 1:step, :] = 0
			elif direction[0] > 0:
				shifted[0:step, :] = 0
		
		elif direction[1] == 1:
			if direction[0] < 0:
				shifted[:, shifted.shape[0] - 1:step] = 0
			elif direction[0] > 0:
				shifted[:, 0:step] = 0
		return shifted
	
	def child(self, array):
		c1 = self.shift(array, (-1, 1)), (-1, 0)
		c2 = self.shift(array, (1, 1)), (1, 0)
		c3 = self.shift(array, (-1, 0)), (0, -1)
		c4 = self.shift(array, (1, 0)), (0, 1)
		return [c1, c2, c3, c4]

	def movement(self):
		front = [[self.cal_error(self.new_img), self.new_img, [0, 0]]]
		print(self.cal_error(self.new_img))
		expanded = []
		expanded_nodes = 0
		print("a* is starting the search")
		while front:
			i = 0
			for j in range(1, len(front)):
				if front[i][0] < front[j][0]:
					i = j
			path = front[i]
			error = front[i][0]
			front = front[:i] + front[i + 1:]
			endnode = path[-2]
			vector = path[-1]
			
			if error > self.thresh:
				break
			if expanded_nodes > 100:
				return None
			if endnode.tolist() in expanded:
				continue
			for k, m in self.child(endnode):
				if k.tolist() in expanded:
					continue
				newpath = [path[0] + self.cal_error(k) - self.cal_error(endnode)] + \
				          path[1:-1] + [k] + [[path[-1][0] + m[0], path[-1][1] + m[1]]]
				front.append(newpath)
			expanded.append(endnode.tolist())
			expanded_nodes += 1
		# print("Expanded nodes:", expanded_nodes)
		# print("Solution:")
		return vector, expanded_nodes
	
	def movement2(self):
		base_points, new_points = self.create_feature()
		print("Error_start:", self.cal_error(self.new_img))
		base_points = base_points.T
		new_points = new_points.T
		base_center = np.array([np.mean(base_points[0]), np.mean(base_points[1])])
		new_center = np.array([np.mean(new_points[0]), np.mean(new_points[1])])
		x, y = (base_center - new_center) * self.scale
		return round(x), round(y)
	
	@staticmethod
	def draw_point_cloud(pre_data, size, color=255, thickness=1):
		zeros = np.zeros(size, np.uint8)
		for x, y in pre_data:
			cv2.line(zeros, (int(x), int(y)), (int(x), int(y)), color, thickness)
		return zeros
	
	def create_feature(self, thresh=15, draw=True):
		good, pts1, pts2, kp1, kp2 = Feature(self.base_img, self.new_img, self.scale).main()
		print("Points:", len(pts1))
		points1, points2 = Filter(pts1, pts2, thresh).filter_points()
		print("Points:", points1.shape[0])
		if draw:
			self.base_img = self.draw_point_cloud(points1,
			                                      (self.base_img.shape[0] // self.scale,
			                                       self.base_img.shape[1] // self.scale))
			self.new_img = self.draw_point_cloud(points2,
			                                     (self.new_img.shape[0] // self.scale,
			                                      self.new_img.shape[1] // self.scale))
			cv2.imshow("Image", cv2.resize(np.hstack((self.base_img, self.new_img)), (1000, 500)))
			key = cv2.waitKey(1)
			if key == ord('q'):
				print("END")
				cv2.destroyWindow("Image")
		return points1, points2
	
	def motion_detection(self, res, thresh):
		"""
		function to calculate a motion-mask. A zero means Motion and one means no Motion.
		:param res: int() - for the lenght
		:param thresh: float() - the allowed error
		:return: np.array() -  motion masks
		"""
		base = cv2.resize(self.base_img, (res, res))
		new = cv2.resize(self.new_img, (res, res))
		d = np.abs(base - new)
		mask = d <= thresh
		return mask
		

if __name__ == '__main__':
	base_img = np.array([[0, 0, 0, 0, 0],
	                     [1, 0, 0, 0, 1],
	                     [1, 1, 1, 1, 1],
	                     [0, 0, 0, 0, 0],
	                     [0, 0, 0, 0, 0]])
	new_img = np.array([[0, 0, 0, 0, 0],
	                    [0, 0, 0, 0, 0],
	                    [0, 1, 1, 1, 0],
	                    [1, 0, 0, 0, 1],
	                    [1, 1, 1, 1, 1]])
	
	mo = Motion(base_img, new_img).motion_detection(5, 0)
	print(mo)
	u_img = np.putmask(base_img, mo == False, 0)
	print(base_img)
