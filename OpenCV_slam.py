import cv2
import numpy as np
from rplidar import RPLidar as rp
import time


def get_coords(g, d):
	x = int(round(np.cos(g * np.pi / 180) * d, 1))
	y = int(round(np.sin(g * np.pi / 180) * d, 1))
	return x, y


def map_coords(points, position):
	return points[0] + position[0], points[1] + position[1]


lidar = rp("/dev/tty.SLAB_USBtoUART")
lidar.connect()
lidar.start_motor()
print(lidar.get_info())
time.sleep(2)
karte = np.zeros((7000, 7000))
position = (3500, 3500)

try:
	counter = 0
	for i, scan in enumerate(lidar.iter_scans()):
		print(i)
		print(scan[0])
		if i % 3 == 1:
			counter = 0
			karte = np.zeros((7000, 7000))
		for _, grad, distance in scan:
			print(grad, distance)
			points = get_coords(grad, distance)
			coords = map_coords(points, position)
			cv2.line(karte, position, coords, 135)
		if i % 3 == 1:
			print(counter)
			resized = cv2.resize(karte, (400, 400))
			cv2.imshow("Karte", resized)
			key = cv2.waitKey(1)
			if key == ord('q'):
				cv2.destroyAllWindows()
				break
			elif key == ord('s'):
				cv2.imwrite("img{i}.jpg".format(i=i), karte)
		counter += 1



finally:
	lidar.stop()
	lidar.stop_motor()
	lidar.disconnect()
	exit()
