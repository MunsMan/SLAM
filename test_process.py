from multiprocessing import Process, Queue, Lock
import time
import numpy as np


def create_data(d_queue):
	for i in range(100):
		d_queue.put((i, np.random.randint(0, 1, 100)))
		time.sleep(0.01)


def use_data(d_queue, a_queue):
	i, d = d_queue.get()
	a_queue.put(d)
	time.sleep(1)
	print(i)
	exit()


data_queue = Queue()
answer_queue = Queue()

pd = Process(target=create_data, args=(data_queue,))

pd.start()
while True:
	if not data_queue.empty():
		Process(target=use_data, args=(data_queue, answer_queue)).start()
