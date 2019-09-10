import numpy as np
import multiprocessing as mp

arr = mp.Array("I", 9)

arr[:] = np.reshape(np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]]), -1)


def sq(arr):
	narr = np.frombuffer(arr.get_obj(), dtype=np.uint32) ** 2
	print(np.reshape(narr, (3, 3)))
	arr[:] = narr


def sq2(arr):
	narr = np.frombuffer(arr.get_obj(), dtype=np.uint32) ** 2
	print(narr)


for i in range(5):
	p = mp.Process(target=sq, args=(arr,))
	d = mp.Process(target=sq2, args=(arr,))
	p.start()
	d.start()
	p.join()
	d.join()
