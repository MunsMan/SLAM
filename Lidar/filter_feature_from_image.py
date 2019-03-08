import cv2
from modules import Feature, Matches
from Filter import Filter

path_img1 = "/Users/hendrikmunske/SLAM/images/img196.jpg"
path_img2 = "/Users/hendrikmunske/SLAM/images/img214.jpg"

img1 = cv2.imread(path_img1)
img2 = cv2.imread(path_img2)

feature = Feature(img1, img2)

kp1, des1, kp2, des2 = feature.get_feat()

good, points1, points2 = feature.match_feat(kp1, des1, kp2, des2)

for i in range(0, 100, 10):
	p1, p2 = Filter(points1, points2, i).filter()
	
	draw_params = dict(matchColor=(0, 255, 0),
	                   singlePointColor=None,
	                   flags=2)
	
	img = Matches(feature.img1, p1, feature.img2, p2).match_drawer()
	
	cv2.imshow("img{}".format(i), img)
if cv2.waitKey(0) == ord('q'):
	cv2.destroyAllWindows()
	exit()
