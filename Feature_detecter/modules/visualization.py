import cv2


class Visualization:
	def __init__(self, q, size=(1000, 500)):
		self.__buffer = q
		self.size = size
	
	def imshow(self):
		c = 0
		cv2.namedWindow("Karte", cv2.WINDOW_AUTOSIZE)
		while True:
			if not self.__buffer.empty():
				img1 = self.__buffer.get()
				print("_______________________")
				cv2.imshow("Karte", cv2.resize(img1, (1000, 500)))
				key = cv2.waitKey(1)
				if key == ord('q'):
					print("END")
					cv2.destroyAllWindows()
					return False
				if key == ord('s'):
					cv2.imwrite("img{i}.jpg".format(i=c), img1)
				c += 1
	
	def main(self):
		
		exec(self.imshow())
