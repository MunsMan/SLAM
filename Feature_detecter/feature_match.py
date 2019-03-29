import numpy as np
import cv2
from modules import Feature, Stitch, Matches, Filter, ClosedPoint

img1 = cv2.imread("img73.jpg")
img2 = cv2.imread("img98.jpg")

print(img1.shape)

feat = Feature(img1, img2, 10)

good, points1, points2, kp1, kp2 = feat.main()

points1, points2 = Filter(points1, points2, 0).filter_points()

matches = Matches(feat.img1, points1, feat.img2, points2)

img3 = matches.match_drawer()

cv2.imshow("img", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
