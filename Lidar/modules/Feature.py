import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import math


class Feature:
	def __init__(self, img1, img2):
		self.img1 = cv2.resize(img1, (1000, 1000))
		self.gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
		
		# img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
		self.img2 = cv2.resize(img2, (1000, 1000))
		self.gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
		
		self.orb = cv2.ORB_create()
	
	def get_feat(self):
		kp1, des1 = self.orb.detectAndCompute(self.gray1, None)
		kp2, des2 = self.orb.detectAndCompute(self.gray2, None)
		
		return kp1, des1, kp2, des2
	
	def draw_feat(self, kp1, kp2, color):
		for points in kp1:
			x, y = points.pt
			self.img1[int(x), int(y)] = color
		
		for points in kp2:
			x, y = points.pt
			self.img2[int(x), int(y)] = color
	
	def match_feat(self, kp1, des1, kp2, des2):
		
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1, des2, 2)
		good = []
		
		# extracting values for the rotation
		points1 = []
		points2 = []
		
		for m, n in matches:
			if m.distance < 0.85 * n.distance:
				good.append(m)
				points1.append(kp1[m.queryIdx].pt)
				points2.append(kp2[m.trainIdx].pt)
		
		return good, points1, points2
	
	def get_rotation(self, points1, points2):
		winkel1 = math.atan(self.cal_line_angle(points1, True, self.img1)) * 180 / math.pi
		winkel2 = math.atan(self.cal_line_angle(points2, True, self.img2)) * 180 / math.pi
		
		winkel = 360 - (winkel1 - winkel2)
		
		return winkel
	
	@staticmethod
	def cal_line_angle(pnt_list, grapf, map):
		X = []
		Y = []
		for x, y in pnt_list:
			X.append(x)
			Y.append(y)
		reg = LinearRegression(fit_intercept=True, n_jobs=-1).fit(np.array(X).reshape(-1, 1),
		                                                          np.array(Y).reshape(-1, 1))
		pred = reg.predict([[1], [2]])
		rotation = ((pred[1] - pred[0]) / 1)
		print("Durchschnittsteiugung:", rotation)
		
		if grapf:
			cv2.line(map, (int(map.shape[0]), int(reg.predict([[map.shape[0]]]))),
			         (int(0), int(reg.predict([[0]]))), (0, 255, 0))
		return rotation


class Stitch:
	
	def __init__(self, img1, img2, good, kp1, kp2):
		self.img1 = img1
		self.img2 = img2
		self.good = good
		self.kp1 = kp1
		self.kp2 = kp2
	
	def stitch(self, points1, points2):
		src_pts = np.float32([points1]).reshape(-1, 1, 2)
		dst_pts = np.float32([points2]).reshape(-1, 1, 2)
		
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		
		print(M)
		
		h, w, _ = self.img1.shape
		pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
		dst = cv2.perspectiveTransform(pts, M)
		img2 = cv2.polylines(self.img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
		
		return img2
