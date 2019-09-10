from modules import Client, Lidar
import numpy as np
from multiprocessing import Queue, Process
import time

buffer = Queue()
lidar = Lidar(buffer)
client = Client()

print(client.setup())
print(client.connect())
print(lidar.start_lidar())

scan_p = Process(target=lidar.get_scan)
scan_p.start()

try:
	while True:
		if not buffer.empty():
			i, scan = buffer.get()
			arr = np.zeros(scan.size + 1, dtype=np.int64)
			arr[0] = i
			arr[1:] = scan.reshape(-1)
			print(arr.dtype)
			bytearr = arr.tobytes()
			client.send(bytearr)
		time.sleep(0.05)

finally:
	scan_p.terminate()
	lidar.stop()
	client.disconnect()
