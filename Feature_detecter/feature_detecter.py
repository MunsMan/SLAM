import cv2
import numpy as np
import os
import sys

img1 = cv2.imread('img196.jpg', 0)
img2 = cv2.imread('img214.jpg', 0)

img1_reshape = cv2.resize(img1, (100, 100))
img2_reshape = cv2.resize(img2, (100, 100))

ps1 = np.isin((img1_reshape > 1), True)
ps2 = np.isin((img2_reshape > 1), True)

points1 = np.where(ps1)
points2 = np.where(ps2)

points1 = np.vstack((points1[0], points1[1]))
points2 = np.vstack((points2[0], points2[1]))

print(points1[0], points2[0])
print("\n")
print(points1[1], points2[1])
print(points2)
print(points1.shape, points2.shape)

comps = np.where(np.isin((points1 == points2), True))

img = np.hstack((img1_reshape, img2_reshape))

cv2.imshow("img", img)

key = cv2.waitKey(0)

if key == ord('q'):
	cv2.destroyAllWindows()
	exit()

if key == ord('s'):
	cv2.imwrite("filter_img.jpg", img)
	cv2.destroyAllWindows()
	exit()
