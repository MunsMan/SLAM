import numpy as np
import cv2
from modules import Feature, Matches, Filter, Motion, Lidar, LidarFunktions, Visualization
from imutils import rotate
import time
from multiprocessing import Process, Queue


def main_loop(lidar_data, lf, image_queue):
	rotation = 0
	size = (5000, 5000)
	mainMap = np.zeros((size), np.uint8)
	lastMap = None
	position = (size[0] // 2, size[1] // 2)
	time.sleep(1)
	print("Starting Main Loop")
	
	while True:
		if not lidar_data.empty():
			st = time.time()
			lineMap = np.zeros(size, np.uint8)
			_, data = lidar_data.get()
			pre_data = lf.prepare_data(data, position)
			# print("Pre_data:", pre_data.shape)
			lt = time.time()
			lineMap = lf.draw_line_map(lineMap, pre_data)
			print("LineMap:", time.time() - lt)
			if lastMap is None:
				mainMap = lf.draw_main_map(pre_data, position, size, None, 0)
			else:
				rt = time.time()
				matches_num, grad, _ = cal_rot_move(lastMap, lineMap, 20)
				print("cal_ro", time.time() - rt)
				rotation += grad
				mt = time.time()
				mainMap = lf.draw_main_map(pre_data, position, size, mainMap, rotation)
				print("MainMap:", time.time() - mt)
			lastMap = lineMap
			print(rotation)
			# feat = Feature(lineMap, 5)
			# lineMap, points = feat.get_corners()
			qt = time.time()
			image_queue.put((mainMap, lineMap))
			print("qt:", time.time() - qt)
			print("Time:", time.time() - st)


def cal_rot_move(img1, img2, thresh):
	feat = Feature(img1, img2, 10)
	
	good, points1, points2, kp1, kp2 = feat.main()
	
	points1, points2 = Filter(points1, points2, thresh).filter_points()
	
	num_matches = points1.shape[0]
	
	print("Matches:", num_matches)
	
	matches = Matches(feat.img1, points1, feat.img2, points2)
	
	motion = Motion(points1, points2)
	
	dgrad, line1, line2 = motion.rotation()
	
	cv2.line(feat.img1, line1[0], line1[1], (0, 0, 255))
	cv2.line(feat.img2, line2[0], line2[1], (0, 255, 0))
	
	print("Grad", dgrad[0, 0])
	
	feat.img2 = rotate(feat.img2, dgrad, (250, 250))
	
	img3 = matches.match_drawer()
	img4 = cv2.add(feat.img1, feat.img2)
	
	_img5 = np.hstack((img3, img4))
	
	return num_matches, dgrad[0, 0], _img5


lidar_data = Queue()
image_queue = Queue()

lidar = Lidar(lidar_data)
lf = LidarFunktions()
print(lidar.start_lidar())
get_lidar_data = Process(target=lidar.get_scan)
main_process = Process(target=main_loop, args=(lidar_data, lf, image_queue))

try:
	get_lidar_data.start()
	main_process.start()
	c = 0
	while True:
		if not image_queue.empty():
			img11, img12 = image_queue.get()
			img1 = np.hstack((img11, img12))
			print(img1.shape)
			print("_______________________")
			cv2.imshow("Karte", cv2.resize(img1, (1000, 500)))
			key = cv2.waitKey(1)
			if key == ord('q'):
				print("END")
				cv2.destroyAllWindows()
				break
			if key == ord('s'):
				cv2.imwrite("img{i}.jpg".format(i=c), img1)
			c += 1
	get_lidar_data.terminate()
	main_process.terminate()

finally:
	lidar.stop()
	print("exit")
	exit()
