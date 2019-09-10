import paho.mqtt.client as mqtt
import time


class Client:
	def __init__(self):
		self.client = mqtt.Client(client_id="RaspberryPi")
		self.ip = "192.168.1.182"
		self.port = 1883
	
	def setup(self):
		self.client.on_connect = self.on_connect
		self.client.on_disconnect = self._on_disconnect
		self.client.on_publish = self._on_publish
		self.client.on_message = self._on_message
		time.sleep(0.1)
		print("Client ready")
	
	@staticmethod
	def on_connect(client, userdata, flags, rc):
		print(client, userdata, flags, rc)
	
	@staticmethod
	def _on_disconnect(client, userdata, rc):
		print(client, userdata, rc)
	
	@staticmethod
	def _on_message(client, userdata, message):
		print(client, userdata, message)
	
	@staticmethod
	def _on_publish(client, userdata, mid):
		print("Message sent: ", mid)
	
	def connect(self):
		self.client.connect(self.ip, self.port)
		print("Connected")
	
	def disconnect(self):
		self.client.disconnect()
		print("disconnected")
	
	def reconnect(self):
		self.client.reconnect()
	
	def send(self, msg):
		self.client.publish("data/lidar", msg)
	
	def main(self):
		self.setup()
		self.connect()


if __name__ == '__main__':
	Server().main()
