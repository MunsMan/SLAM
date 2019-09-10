import pandas as pd
import numpy as np
import cv2
from modules import lf, icp
import time

df = pd.read_csv("data/data_set.csv")

data_set1 = np.array([df["0"], df["1"]], dtype=np.int64).T
data_set2 = np.array([df["2"], df["3"]], dtype=np.int64).T

lf = lf()

data_set2_t = data_set2 + np.array([-100, 80])

s = time.time()
R, T = icp(data_set1.T[:, :500], data_set2_t.T[:, :500])
print("ICP-Time:", time.time() - s)

print(R, T)

data_set2 = np.array(np.dot(data_set2_t, R) + T, dtype=np.int64)

pc1 = lf.draw_point_cloud(data_set1, (5000, 5000))
pc2 = lf.draw_point_cloud(data_set2_t, (5000, 5000))
pc3 = lf.draw_point_cloud(data_set2, (5000, 5000))

image3 = cv2.add(pc1, pc3)
images = cv2.add(pc1, pc2)

cv2.imshow("Images", cv2.resize(images, (800, 800)))
cv2.imshow("Result", cv2.resize(image3, (1000, 1000)))
cv2.waitKey(0)
cv2.destroyAllWindows()
