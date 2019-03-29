import numpy as np
from matplotlib import pyplot as plt


class ClosedPoint:
	
	def __init__(self, points1, points2, error=0.0001, maxiter=100):
		self.ae = error
		self.maxiter = maxiter
		self.points1 = np.array([points1[:, 0], points1[:, 1]])
		self.points2 = np.array([points2[:, 0], points2[:, 1]])
	
	def inter_closed_point(self):
		
		H = None
		dError = 1000.0
		preError = 1000.0
		counter = 0
		
		while dError >= self.ae:
			plt.cla()
			plt.plot(self.points1[0, :], self.points1[1, :], ".r")
			plt.plot(self.points2[0, :], self.points2[1, :], ".b")
			plt.axis("equal")
			plt.show()
			inds, error = self.nearest_neighbor_as(self.points1, self.points1)
			print(inds)
			Rt, Tt = self.estimate_motion(self.points1[:, inds], self.points2)
			
			# update current points
			self.points2 = (Rt @ self.points2) + Tt[:, np.newaxis]
			
			H = self.update_homogeneous_matrix(H, Rt, Tt)
			
			dError = abs(preError - error)
			preError = error
			print("Residual:", error)
			
			if dError <= self.ae:
				print("Converge", error, dError, counter)
				break
			elif self.maxiter <= counter:
				print("Not Converge...", error, dError, counter)
				break
		
		R = np.array(H[0:2, 0:2])
		T = np.array(H[0:2, 2])
		
		return H, self.points2
	
	@staticmethod
	def update_homogeneous_matrix(Hin, R, T):
		H = np.zeros((3, 3))
		
		H[0, 0] = R[0, 0]
		H[1, 0] = R[1, 0]
		H[0, 1] = R[0, 1]
		H[1, 1] = R[1, 1]
		H[2, 2] = 1.0
		
		H[0, 2] = T[0]
		H[1, 2] = T[1]
		
		if Hin is None:
			return H
		else:
			return Hin @ H
	
	@staticmethod
	def nearest_neighbor_as(points1, points2):
		# calc the sum of residual errors
		dcpoints = points1 - points2
		d = np.linalg.norm(dcpoints, axis=0)
		error = sum(d)
		
		# calc index with nearest neighbor assosiation
		inds = []
		for i in range(points2.shape[1]):
			minid = -1
			mind = float("inf")
			for ii in range(points1.shape[1]):
				d = np.linalg.norm(points1[:, ii] - points2[:, i])
				
				if mind >= d:
					mind = d
					minid = ii
			
			inds.append(minid)
		
		return inds, error
	
	@staticmethod
	def estimate_motion(points1, points2):
		pm = np.mean(points1, axis=1)
		cm = np.mean(points2, axis=1)
		
		shift1 = points1 - pm[:, np.newaxis]
		shift2 = points2 - cm[:, np.newaxis]
		
		W = shift2 @ shift1.T
		u, s, vh = np.linalg.svd(W)
		
		R = (u @ vh).T
		t = pm - (R @ cm)
		
		return R, t
