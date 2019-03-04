import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from imutils import rotate
import math
import time
from modules import Feature, Stitch


def color_from_list(img, points, color):
	for x, y in points:
		img = cv2.rectangle(img, (int(x) - 1, int(y) - 1), (int(x) + 1, int(y) + 1), color, thickness=1)
	return img


img1 = cv2.imread('img196.jpg')
img2 = cv2.imread('img214.jpg')

st = time.time()

feature = Feature(img1, img2)

kp1, des1, kp2, des2 = feature.get_feat()

good, points1, points2 = feature.match_feat(kp1, des1, kp2, des2)

# winkel = feature.get_rotation(points1, points2)

# img2 = rotate(img2, winkel)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None)

stitched_img = Stitch(feature.img1, feature.img2, good, kp1, kp2).stitch(points1, points2)

if False in (stitched_img == feature.img2):
	print("not the same")
else:
	print("Same")

stitched_img = color_from_list(stitched_img, points2, (0, 255, 0))
stitched_img = color_from_list(stitched_img, points1, (255, 0, 0))

print(stitched_img.shape)

print(time.time() - st)

cv2.imshow("stich", stitched_img)

# cv2.imshow("img", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
