import numpy as np


class Filter:
	
	def __init__(self, pts1, pts2, thresh):
		self.p1 = np.array(pts1)
		self.p2 = np.array(pts2)
		self.thresh = thresh
	
	@staticmethod
	def cal(points):
		pa = []
		for i in range(points.shape[1]):
			p = []
			for x in range(points.shape[0]):
				dx = abs(np.sum(np.abs(points[:, i] - points[x, i])) / (points.shape[0] - 1))
				p.append([dx])
			p = abs(p - np.mean(p))
			pa.append(p)
		
		return np.array(pa)
	
	def filter_points(self):
		error = np.sum(np.abs(self.cal(self.p1) - self.cal(self.p2)))
		while error > self.thresh:
			dl = np.abs(self.cal(self.p1) - self.cal(self.p2))
			self.p1 = np.delete(self.p1, np.where(np.isin(dl, np.max(dl)))[1], 0)
			self.p2 = np.delete(self.p2, np.where(np.isin(dl, np.max(dl)))[1], 0)
			error = np.sum(np.abs(self.cal(self.p1) - self.cal(self.p2)))
			print(error)
		return self.p1, self.p2
