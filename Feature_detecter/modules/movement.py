import numpy as np
from sklearn.linear_model import LinearRegression


class Motion:
	
	def __init__(self, pts1, pts2, scale):
		self.pts1 = pts1
		self.pts2 = pts2
		self.scale = 5000 // scale
	
	def rotation(self):
		pointmap1 = LinearRegression(fit_intercept=True, n_jobs=-1)
		pointmap2 = LinearRegression(fit_intercept=True, n_jobs=-1)
		
		pointmap1.fit(self.pts1[:, 0].reshape(-1, 1), self.pts1[:, 1].reshape(-1, 1))
		pointmap2.fit(self.pts2[:, 0].reshape(-1, 1), self.pts2[:, 1].reshape(-1, 1))
		
		grad1 = np.round(np.arctan(pointmap1.coef_) / (2 * np.pi) * 360, 2)
		grad2 = np.round(np.arctan(pointmap2.coef_) / (2 * np.pi) * 360, 2)
		
		dgrad = (grad2 - grad1)
		
		line1 = (0, pointmap1.predict(0)), (self.scale, pointmap1.predict(500))
		line2 = (0, pointmap2.predict(0)), (self.scale, pointmap2.predict(500))
		
		return dgrad, line1, line2
		
	def move(self):
		pass
	
	def movement(self):
		pass
