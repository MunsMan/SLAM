import numpy as np
import cv2
from .modules import icp as icp_algo


class Map:
	def __init__(self):
		pass
	
	@staticmethod
	def icp(base, new_set, amount_of_points=100, max_iter=10, eps=1, show_animation=False):
		set1 = base.T[:, :amount_of_points]
		set2 = new_set.T[:, :amount_of_points]
		R, T = icp_algo(set1, set2, eps=eps, max_iter=max_iter, show_animation=show_animation)
		return R, T
	
	@staticmethod
	def transform_set(set, trans):
		R, T = trans
		return np.array(np.dot(set, R) + T, dtype=np.int64)
