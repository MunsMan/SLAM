import cv2
import numpy as np
from modules import Lidar, ClosedPoint, Feature
from imutils import rotate
from multiprocessing import Process

img1 = cv2.imread("img196.jpg")
img2 = cv2. imread("img214.jpg")

img2 = rotate(img2, 20)

mainMap = np.zeros((7000, 7000))
car_position = (3500, 3500)

feature = Feature(img1, img2)
closed_point = ClosedPoint()

kp1, des1, kp2, des2 = feature.get_feat()

good, points1, points2 = feature.match_feat(kp1, des1, kp2, des2)


R, T, points2 = closed_point.closest_point_matching(points1, points2)

for x, y in points2:
	print(x, y)
	cv2.rectangle(feature.img1, (int(x), int(y)), (int(x), int(y)), 255, 1)
	
for x, y in points2:
	print(x, y)
	cv2.rectangle(feature.img2, (int(x), int(y)), (int(x), int(y)), 255, 1)

cv2.imshow("img", np.hstack((feature.img1, rotate(feature.img2, -20))))
if cv2.waitKey(0) == ord('q'):
	cv2.destroyAllWindows()
	exit()
