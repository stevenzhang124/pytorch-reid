import requests
import subprocess
from flask import Flask, request
#from reid import cam_reid
#import psutil  #pip3 install psutil
#import GPUtil
import json
import numpy as np
import cv2
import sys
from concurrent.futures import ThreadPoolExecutor
import time

#video_stream = []
executor = ThreadPoolExecutor(3)

app = Flask(__name__)

def open_cam_rtsp(uri, width, height, latency):
	gst_str = ('rtspsrc location={} latency={} ! '
			   'rtph264depay ! h264parse ! omxh264dec ! '
			   'nvvidconv ! '
			   'video/x-raw, width=(int){}, height=(int){}, '
			   'format=(string)BGRx ! '
			   'videoconvert ! appsink').format(uri, latency, width, height)
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


@app.route('/controller', methods=['GET', 'POST'])
def controller():
	#global video_stream

	#parse the message from server
	messages = request.form.to_dict()
	print(messages)

	if messages['uri']:
		video_stream = messages['uri']

	if messages['kill-uri']:
		video_stream.remove(messages['kill-uri'])

	if messages['kill-detection']:
		info = requests.get('http://localhost:5001/kill-detection')

	if messages['kill-reid']:
		info = requests.get('http://localhost:5002/kill-reid')

	if messages['start-detection']:
		cmd = "python3 worker-detection.py"
		subprocess.Popen(cmd, shell=True)

	if messages['start-reid']:
		cmd = "python3 worker-reid.py"
		subprocess.Popen(cmd, shell=True)

	if messages['offload-url']:
		offload_url = messages['offload-url']

	#send frames
	#for uri in video_stream:
	if video_stream:
		uri = "rtsp://admin:edge1234@" + video_stream + ":554/cam/realmonitor?channel=1&subtype=0"
		#print(uri)
		executor.submit(read_video, uri, offload_url)
	return ''

	
#def read_video(uri, offload_url):
def read_video(vid):
	#get the video ID
	#uri = "rtsp://admin:edge1234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
	t1 = time.time()
	#cap = open_cam_rtsp(uri, 640, 480, 200)
	cap = cv2.VideoCapture('./test_img/' + str(vid) + '.mp4')
	t2 = time.time()
	print("read consumers", t2-t1)
	


	if not cap.isOpened():
		sys.exit('Failed to open camera!')


	counter=0
	images = 0
	t_total = 0
	while (cap.isOpened()):
		#t1 = cv2.getTickCount()
		counter+=1
		#if counter % 12 !=0:
		#	print(counter)
		#	continue
		
		if counter % 5 != 0:
			ret, frame = cap.read()
			continue
		#break
	
		file = {"file": ("file_name.jpg", cv2.imencode(".jpg", frame)[1].tobytes(), "image/jpg")}
		data = {}
		offload_url = '192.168.1.109'
		data['offload-url'] = offload_url
		t1 = time.time()
		info = requests.post('http://localhost:5001/detection', data=data, files=file)
		t2 = time.time()
		print("send consumes", t2-t1)
		images += 1
		t_total = t_total + float(info.text)
		if t_total > 3.0 :
			print('estimated fps for ' + str(vid) +'is', images/3)
			break
		#print(uri)
		#print("send once")
		


# @app.route('/get_status', methods=['GET'])
# def get_status():
# 	cpu_usage = psutil.cpu_percent()
# 	#print('available memory：%dM %fG'%(mem.available//M, mem.available/G))
# 	mem = psutil.virtual_memory()
# 	mem_usage = mem.available
# 	Gpus = GPUtil.getGPUs()
# 	for gpu in Gpus:
# 		gpu_usage = gpu.memoryUtil * 100

# 	info = {}
# 	info['node'] = '192.168.1.100'
# 	info['cpu_usage'] = cpu_usage
# 	info['mem_usage'] = mem_usage
# 	info['gpu_usage'] = gpu_usage

# 	print(info)

# 	return json.dumps(info)


if __name__ == '__main__':
	#app.run(host='0.0.0.0', port=5000, debug=False, threaded = True)
	# uri = "rtsp://admin:edge1234@192.168.1.117:554/cam/realmonitor?channel=1&subtype=0"
	# offload_url = '192.168.1.103'
	# read_video(uri, offload_url)
	vids = [1, 3, 5]
	for vid in vids:
		executor.submit(read_video, vid)



