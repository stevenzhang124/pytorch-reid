#this file is used to test the time needed to transmit an image

import requests
import cv2
import time

def send_request():
	img = cv2.imread('example.jpg')
	for i in range(5):

		file = {"file": ("file_name.jpg", cv2.imencode(".jpg", img)[1].tobytes(), "image/jpg")}
		data = {}
		data['offload-url'] = '192.168.1.103'
		t1 = time.time()
		info = requests.post('http://192.168.1.103:5001/detection', data=data, files=file)
		t2 = time.time()
		print("send consumes", t2-t1)
		print(info.text)

send_request()