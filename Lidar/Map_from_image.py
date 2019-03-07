import cv2
import numpy as np
from modules import Lidar, ClosedPoint, Feature
from imutils import rotate
from multiprocessing import Process, Queue


def match(queue, img1, img2):
	feature = Feature(img1, img2)
	closed_point = ClosedPoint()
	
	kp1, des1, kp2, des2 = feature.get_feat()
	
	good, points1, points2 = feature.match_feat(kp1, des1, kp2, des2)
	
	R, T, points2 = closed_point.closest_point_matching(points1, points2)
	
	queue.put((R, T, points2))


img1 = cv2.imread("img196.jpg")
img2 = cv2.imread("img214.jpg")

queue = Queue()

img21 = rotate(img2, 90)
img22 = rotate(img2, 180)
img23 = rotate(img2, 270)

p1 = Process(match, args=(queue, img1, img2))
p2 = Process(match, args=(queue, img1, img21))
p3 = Process(match, args=(queue, img1, img22))
p4 = Process(match, args=(queue, img1, img23))

p1.start()
p2.start()
p3.start()
p4.start()

mainMap = np.zeros((7000, 7000))
car_position = (3500, 3500)

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
