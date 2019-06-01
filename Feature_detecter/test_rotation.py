import cv2
import numpy as np
from imutils import rotate

img1 = cv2.imread("lineMap31.jpg", 0)
img2 = cv2.imread("lineMap31.jpg", 0)

lines = cv2.HoughLinesP(image=img1, rho=0.02, theta=np.pi / 500, threshold=10, lines=np.array([]), maxLineGap=100)

lines_len = np.square(lines[:, 0, 0] - lines[:, 0, 2] + lines[:, 0, 1] - lines[:, 0, 3])
pos_max = np.where(np.isin(lines_len, np.max(lines_len)))
obj = lines[pos_max, 0]
x1, y1, x2, y2 = obj.reshape(-1)

img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

print(img.shape)

cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow("Test", cv2.resize(img, (400, 400)))
cv2.waitKey(0)
cv2.destroyAllWindows()
