import multiprocessing as mp
import numpy as np
import ctypes as c
import time


def cal_square(p_id, array):
	array = np.array(np.frombuffer(array.get_obj(), np.uint8).reshape((10, 10)), dtype=np.uint16)
	for x in range(len(array)):
		for y in range(len(array)):
			array[x, y] = array[x, y] ** 2
	print(p_id, array[:])
	time.sleep(2)


data = np.arange(0, 100, 1).reshape((10, 10))

arr = mp.Array(c.c_int8, data.shape[0] * data.shape[1])

new_array = np.frombuffer(arr.get_obj(), np.uint8).reshape((10, 10))

arr[:] = data[:].reshape(-1)

p1 = mp.Process(target=cal_square, args=(1, arr,))
p2 = mp.Process(target=cal_square, args=(2, arr,))
p3 = mp.Process(target=cal_square, args=(3, arr,))
p4 = mp.Process(target=cal_square, args=(4, arr,))

p1.start()
p2.start()
p3.start()
p4.start()
p1.join()
p2.join()
p3.join()
p4.join()
print("Last")
