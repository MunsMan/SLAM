import numpy as np
import math
import cv2


class Motion:
	
	def __init__(self, base_img, new_img, scale=10):
		self.scale = 5000 // scale
		self.base_img = cv2.resize(base_img, (self.scale, self.scale))
		self.base_center = self.center(self.base_img)
		self.new_img = cv2.resize(new_img, (self.scale, self.scale))
		self.weight = 0.00001
		self.thresh = 0.45
	
	def cal_error(self, new_img):
		b_base_img = np.isin(self.base_img, np.max(self.base_img))
		b_new_img = np.isin(new_img, np.max(new_img))
		derror1 = np.sum(np.bitwise_and(b_new_img, b_base_img)) / np.nonzero(b_new_img)[0].shape[0]
		new_img_center = self.center(b_new_img)
		derror2 = math.sqrt(
			(self.base_center[0] - new_img_center[0]) ** 2 + abs(self.base_center[1] - new_img_center[1]) ** 2)
		error = (derror1 + self.weight)  # + 1 / ((derror2 + self.weight)*10)
		return error
	
	@staticmethod
	def center(array):
		pts = np.nonzero(array)
		return np.array([np.mean(pts[0]), np.mean(pts[1])])
	
	def shift(self, array, direction):
		shifted = np.roll(array, direction[0], axis=direction[1])
		if direction[1] == 0:
			if direction[0] < 0:
				shifted[shifted.shape[0] - 1, :] = 0
			elif direction[0] > 0:
				shifted[0, :] = 0
		
		elif direction[1] == 1:
			if direction[0] < 0:
				shifted[:, shifted.shape[0] - 1] = 0
			elif direction[0] > 0:
				shifted[:, 0] = 0
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
			if expanded_nodes > 25:
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


if __name__ == '__main__':
	base_img = np.array([[0, 0, 0, 0, 0],
	                     [1, 0, 0, 0, 1],
	                     [1, 1, 1, 1, 1],
	                     [0, 0, 0, 0, 0],
	                     [0, 0, 0, 0, 0]])
	new_img = np.array([[0, 0, 0, 0, 0],
	                    [0, 0, 0, 0, 0],
	                    [0, 0, 0, 0, 0],
	                    [1, 0, 0, 0, 1],
	                    [1, 1, 1, 1, 1]])
	
	mo = Motion(base_img, new_img)
	print("lu")
	lu = mo.cal_error(mo.shift(mo.new_img, (-1, 1)))
	print("ru")
	ru = mo.cal_error(mo.shift(mo.new_img, (1, 1)))
	print("ld")
	ld = mo.cal_error(mo.shift(mo.new_img, (-1, 0)))
	print("rd")
	rd = mo.cal_error(mo.shift(mo.new_img, (1, 0)))
	
	print(mo.movement())
