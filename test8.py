#import matplotlib
#matplotlib.use('Agg')
from flask import Flask, render_template, Response
import sys
import time
import logging
import subprocess
import cv2
from pedestrian_detection_ssdlite import api
from reid import cam_reid
from matplotlib import pyplot as plt

app = Flask(__name__)

logging.basicConfig(
	stream=sys.stdout,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	datefmt=' %I:%M:%S ',
	level="INFO"
)
logger = logging.getLogger('detector')

'''
def open_cam_onboard(width, height):
	# On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
	gst_str = ('nvcamerasrc ! '
			   'video/x-raw(memory:NVMM), '
			   'width=(int)2592, height=(int)1458, '
			   'format=(string)I420, framerate=(fraction)30/1 ! '
			   'nvvidconv ! '
			   'video/x-raw, width=(int){}, height=(int){}, '
			   'format=(string)BGRx ! '
			   'videoconvert ! appsink').format(width, height)
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
'''

reid_mode = cam_reid.reid_model()

# encode origin image
compare = cam_reid.Compare(model=reid_mode, origin_img="./image/origin")
origin_f, origin_name = compare.encode_origin_image()



def open_cam_onboard(width, height):
	gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
	if 'nvcamerasrc' in gst_elements:
		# On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
		gst_str = ('nvcamerasrc ! '
				   'video/x-raw(memory:NVMM), '
				   'width=(int)2592, height=(int)1458, '
				   'format=(string)I420, framerate=(fraction)30/1 ! '
				   'nvvidconv ! '
				   'video/x-raw, width=(int){}, height=(int){}, '
				   'format=(string)BGRx ! '
				   'videoconvert ! appsink').format(width, height)
	elif 'nvarguscamerasrc' in gst_elements:
		gst_str = ('nvarguscamerasrc ! '
				   'video/x-raw(memory:NVMM), '
				   'width=(int)1920, height=(int)1080, '
				   'format=(string)NV12, framerate=(fraction)30/1 ! '
				   'nvvidconv flip-method=0 ! '
				   'video/x-raw, width=(int){}, height=(int){}, '
				   'format=(string)BGRx ! '
				   'videoconvert ! appsink').format(width, height)
	else:
		raise RuntimeError('onboard camera source not found!')
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def handle_frames(frame):
	detection_results = api.get_person_bbox(frame, thr=0.4)

	bounding_boxs = []
	for bbox in detection_results:
		logger.info('coordinates: {} {}. '.
					format(bbox[0], bbox[1]))

		x1 = int(bbox[0][0])
		y1 = int(bbox[0][1])
		x2 = int(bbox[1][0])
		y2 = int(bbox[1][1])

		person = frame[y1:y2, x1:x2, :]

		identify_name, score = compare.run(person, origin_f, origin_name)

		if(identify_name in [ "MJ1", "MJ2", "MJ3", "MJ4", "MJ5", "MJ6", "MJ7", "MJ8", "MJ9", "MJ10"]):
				identify_name = "Person_1"
		elif(identify_name in ["QY1", "QY2", "QY3", "QY4", "QY5", "QY6", "QY7", "QY8", "QY9", "QY10"]):
			identify_name = "Person_2"
			
		print("identify name:{}, score:{}".format(identify_name, round(1-score, 2)))
		
		bounding_boxs.append([(x1,y1), (x2,y2), identify_name+' '+str(round(1-score, 2))])
		#img = cam_detection.draw_rectangle(img, (x1,y1,x2,y2), identify_name+'  '+str(round((1-score), 2)))
			
	for obj in bounding_boxs:
		print(obj)
		cv2.putText(frame, obj[2], (obj[0][0], obj[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
		frame = cv2.rectangle(frame, obj[0], obj[1], (0, 255, 0), 2)
		
	return frame


def gen_frames():  # generate frame by frame from camera
	#stream detection
	#cap = open_cam_onboard(640, 480)
	#uri = "rtsp://admin:admin@192.168.1.106:554/stream2"
	uri = "rtsp://admin:edge1234@192.168.1.106:554/cam/realmonitor?channel=1&subtype=1"
	cap = open_cam_rtsp(uri, 640, 480, 200)

	if not cap.isOpened():
		sys.exit('Failed to open camera!')

	# allow the camera to warmup
	#time.sleep(0.1)
	frame_rate_calc = 1
	#freq = cv2.getTickFrequency()
	#print(freq)\
	counter=0

	while (cap.isOpened()):
		#t1 = cv2.getTickCount()
		counter+=1
		#if counter % 12 !=0:
		#	print(counter)
		#	continue
		t1 = time.time()
		print ("before read:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
		if counter % 5 != 0:
			ret, frame = cap.read()
			print ("after read", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
			continue

		logger.info("FPS: {0:.2f}".format(frame_rate_calc))
		#cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
		#			cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

		#result = api.get_person_bbox(frame, thr=0.6)  #add functions to this line
		frame = handle_frames(frame)
		print ("after handle", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

		t2 = time.time()
		print("one frame takes {0:.2f}".format(t2-t1))
		frame_rate_calc = 1 / (t2 - t1)
		cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
					cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

		#if counter < 5:
		#	plt.imshow(frame[:, :, ::-1])
		#	plt.show()
		#	continue		

		# show the frame
		cv2.imshow("Stream from EdgeNX1", frame)
		key = cv2.waitKey(1) & 0xFF

		#t2 = cv2.getTickCount()
		
		#time1 = (t2 - t1) / freq
		#frame_rate_calc = 1 / time1
		#print("one frame takes {0:.2f}".format(t2-t1))
		
		#(flag, outputFrame) = cv2.imencode(".jpg", frame)
		#yield (b'--frame\r\n'
		#			   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(outputFrame) + b'\r\n')
		

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

if __name__ == '__main__':
	img = cv2.imread('example.jpg')
	img = handle_frames(img)
	plt.imshow(img[:, :, ::-1])
	print("show frame")
	plt.show()
	gen_frames()
