import numpy as np
import pandas as pd
import cv2
import sklearn
import matplotlib
from modules import LidarFunctions

lf = LidarFunctions()

data_frame = pd.read_csv("data_set.csv")
data_set1 = np.array([data_frame["0"], data_frame["1"]], dtype=np.uint8)
data_set2 = np.array([data_frame["2"], data_frame["3"]], dtype=np.uint8)

print(data_set1)

image1 = lf.draw_point_cloud(data_set1.T, (500, 500))
image2 = lf.draw_point_cloud(data_set2.T, (500, 500))

images = np.hstack((image1, image2))

print(images.shape)

cv2.imshow("Image", cv2.resize(images, (1000, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
