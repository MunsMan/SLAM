import cv2
import numpy as np


class CornerFeature:
	
	def __init__(self, image, scale=0, thresh=0.03, position=(5000, 5000)):
		self.image = image
		self.scale = scale
		self.thresh = thresh
		self.position = position
	
	def get_corners(self):
		image = cv2.resize(self.image,
		                   (self.image.shape[0] // self.scale, self.image.shape[0] // self.scale),
		                   interpolation=cv2.INTER_AREA)
		dst = cv2.cornerHarris(image, 2, 5, 0.05)
		dst = cv2.dilate(dst, None)
		image[dst > self.thresh * dst.max()] = [255]
		pts = np.where(dst > self.thresh * dst.max())
		image = cv2.resize(image,
		                   (image.shape[0] * self.scale, image.shape[0] * self.scale),
		                   interpolation=cv2.INTER_CUBIC)
		return image, pts


class Feature:
	
	def __init__(self, img1, img2, scale):
		self.img1 = cv2.resize(img1,
		                       (img1.shape[0] // scale, img1.shape[1] // scale),
		                       interpolation=cv2.INTER_AREA)
		self.gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
		
		self.img2 = cv2.cv2.resize(img2,
		                           (img2.shape[0] // scale, img2.shape[1] // scale),
		                           interpolation=cv2.INTER_AREA)
		self.gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
		
		self.orb = cv2.ORB_create()
	
	def get_feat(self):
		kp1, des1 = self.orb.detectAndCompute(self.gray1, None)
		kp2, des2 = self.orb.detectAndCompute(self.gray2, None)
		
		return kp1, des1, kp2, des2
	
	def draw_feat(self, kp1, kp2, color):
		for pts in kp1:
			x, y = pts.pt
			self.img1[int(x), int(y)] = color
		
		for pts in kp2:
			x, y = pts.pt
			self.img2[int(x), int(y)] = color
	
	@staticmethod
	def match_feat(kp1, des1, kp2, des2):
		
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
	
	def main(self, draw=False, color=(255, 255, 255)):
		
		kp1, des1, kp2, des2 = self.get_feat()
		good, points1, points2 = self.match_feat(kp1, des1, kp2, des2)
		if draw:
			self.draw_feat(kp1, kp2, color)
		
		return good, points1, points2, kp1, kp2


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


class Matches:
	def __init__(self, img1, points1, img2, points2, color=(255, 0, 0)):
		self.img1 = img1
		self.img2 = img2
		self.p1 = np.array(points1)
		self.p2 = np.array(points2)
		self.c = color
	
	def match_drawer(self):
		img3 = np.hstack((self.img1, self.img2))
		self.p2 = np.array(self.transform_p2())
		if self.p1.shape == self.p2.shape:
			for i in range(self.p1.shape[0] - 1):
				img3 = cv2.line(img3,
				                (int(self.p1[i][0]), int(self.p1[i][1])),
				                (int(self.p2[i][0]), int(self.p2[i][1])),
				                self.c)
			
			return img3
		else:
			print("Shapes don't match.")
			print(self.p1.shape, self.p2.shape)
	
	@staticmethod
	def move_points(points, move):
		return points + move
	
	def transform_p2(self):
		mx, my = self.img1.shape[0], self.img1.shape[1]
		px = self.move_points(self.p2[:, 0], mx)
		py = self.p2[:, 1]
		p2 = np.stack((px[:], py[:]), axis=1)
		return p2
