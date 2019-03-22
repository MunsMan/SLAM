import cv2
import numpy as np
from modules import Feature, Matches, ClosedPoint
from Filter import Filter
from imutils import rotate

path_img1 = "/Users/hendrikmunske/SLAM/images/img196.jpg"
path_img2 = "/Users/hendrikmunske/SLAM/images/img214.jpg"

img1 = cv2.imread(path_img1)
img2 = cv2.imread(path_img2)

img2 = rotate(img2, 20)

feature = Feature(img1, img2)

kp1, des1, kp2, des2 = feature.get_feat()

good, points1, points2 = feature.match_feat(kp1, des1, kp2, des2)

p1, p2 = Filter(points1, points2, 50).filter_points()

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   flags=2)

img = Matches(feature.img1, p1, feature.img2, p2).match_drawer()

H, points22 = ClosedPoint(p1, p2).inter_closed_point()

print(H, points22)

cv2.imshow("img", img)
key = cv2.waitKey(0)
if key == ord('s'):
	cv2.imwrite("img.jpg", img)
	cv2.destroyAllWindows()
	exit()
if key == ord('q'):
	cv2.destroyAllWindows()
	exit()
