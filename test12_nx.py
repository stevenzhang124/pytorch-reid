#import matplotlib
#matplotlib.use('Agg')
from flask import Flask, render_template, Response
import sys
import time
import logging
import subprocess
import cv2

from collections import deque
from tracker import Tracker
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from pedestrian_detection_ssdlite import api
from reid import cam_reid
from matplotlib import pyplot as plt
import mysql.connector

# global variables to be used in the code for tracker
max_age=5
min_hits=1
tmp = 0
tmp2 = 0

app = Flask(__name__)

# logging.basicConfig(
# 	stream=sys.stdout,
# 	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# 	datefmt=' %I:%M:%S ',
# 	level="INFO"
# )
# logger = logging.getLogger('detector')

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

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])


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


def box_iou2(a, b):
	'''
	Helper funciton to calculate the ratio between intersection and the union of
	two boxes a and b
	a[0], a[1], a[2], a[3] <-> left, up, right, bottom
	'''
	
	w_intsec = np.maximum (0, (np.minimum(a[1][0], b[1][0]) - np.maximum(a[0][0], b[0][0])))
	h_intsec = np.maximum (0, (np.minimum(a[1][1], b[1][1]) - np.maximum(a[0][1], b[0][1])))
	s_intsec = w_intsec * h_intsec
	s_a = (a[1][0] - a[0][0])*(a[1][1] - a[0][1])
	s_b = (b[1][0] - b[0][0])*(b[1][1] - b[0][1])
  
	return float(s_intsec)/(s_a + s_b -s_intsec)

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
	'''
	From current list of trackers and new detections, output matched detections,
	unmatchted trackers, unmatched detections.
	'''    
	
	IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
	for t,trk in enumerate(trackers):
		#trk = convert_to_cv2bbox(trk) 
		for d,det in enumerate(detections):
		 #   det = convert_to_cv2bbox(det)
			IOU_mat[t,d] = box_iou2(trk,det) 
	
	# Produces matches       
	# Solve the maximizing the sum of IOU assignment problem using the
	# Hungarian algorithm (also known as Munkres algorithm)
	
	matched_idx_tra, matched_idx_det = linear_assignment(-IOU_mat)        
	matched_idx = np.zeros((len(matched_idx_tra),2),dtype=np.int8)
	for i in range(len(matched_idx_tra)):
		matched_idx[i]=(matched_idx_tra[i],matched_idx_det[i])
	   

	unmatched_trackers, unmatched_detections = [], []
	for t,trk in enumerate(trackers):
		if(t not in matched_idx[:,0]):
			unmatched_trackers.append(t)

	for d, det in enumerate(detections):
		if(d not in matched_idx[:,1]):
			unmatched_detections.append(d)

	matches = []
   
	# For creating trackers we consider any detection with an 
	# overlap less than iou_thrd to signifiy the existence of 
	# an untracked object
	
	for m in matched_idx:
		if(IOU_mat[m[0],m[1]]<iou_thrd):
			unmatched_trackers.append(m[0])
			unmatched_detections.append(m[1])
		else:
			matches.append(m.reshape(1,2))
	
	if(len(matches)==0):
		matches = np.empty((0,2),dtype=int)
	else:
		matches = np.concatenate(matches,axis=0)
	
	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def draw_box_label(img, bbox_cv2, box_color=(0, 255, 255), personReID_info={'personID':'Unknown'}, show_label=True):
	'''
	Helper funciton for drawing the bounding boxes and the labels
	bbox_cv2 = [left, top, right, bottom]
	'''
	#box_color= (0, 255, 255)
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_size = 0.8
	font_color = (0, 0, 255)
	left, top, right, bottom = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
	
	# Draw the bounding box
	cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)
	
	if show_label:
		# Draw a filled box on top of the bounding box (as the background for the labels)
		cv2.rectangle(img, (left-2, top-10), (right+2, top), box_color, -1, 1)
		
		# Output the labels that show the x and y coordinates of the bounding box center.
		text_ID = 'PersonID:' + personReID_info['personID']
		cv2.putText(img,text_ID,(left,top), font, font_size, font_color, 2, cv2.LINE_AA)
		#text_x= 'x='+str((left+right)/2)
		#cv2.putText(img,text_x,(left,top-10), font, font_size, font_color, 1, cv2.LINE_AA)
		#text_y= 'y='+str((top+bottom)/2)
		#cv2.putText(img,text_y,(left,top), font, font_size, font_color, 1, cv2.LINE_AA)
			
	return img    

def db_connection():
	mydb = mysql.connector.connect( host = '192.168.1.100',
	user = 'cqy',
	port = '3306',
	database = 'edgedemo',
	passwd = '123456',
	autocommit = True)
	return mydb

def handle_frames(frame):

	global tracker_list
	global max_age
	global min_hits
	global track_id_list
	global tmp
	global tmp2

	

	detection_results = api.get_person_bbox(frame, thr=0.5)
	
	x_box =[]
	if len(tracker_list) > 0:
		for trk in tracker_list:
			x_box.append([(trk.box[0],trk.box[1]),(trk.box[2],trk.box[3])]) #should be changed into the right format instead of the .box format
			
	matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, detection_results, iou_thrd = 0.2)  
	
	# Deal with matched detections     
	if matched.size >0:
		for trk_idx, det_idx in matched:
			z = detection_results[det_idx]
			z = np.expand_dims([n for a in z for n in a], axis=0).T
			tmp_trk= tracker_list[trk_idx]
			tmp_trk.kalman_filter(z)
			xx = tmp_trk.x_state.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]
			x_box[trk_idx] = xx
			tmp_trk.box =xx
			tmp_trk.hits += 1
			tmp_trk.no_losses = 0
	
	# Deal with unmatched detections      
	if len(unmatched_dets)>0:
		for idx in unmatched_dets:
			z = detection_results[idx]
			x1 = int(z[0][0])
			y1 = int(z[0][1])
			x2 = int(z[1][0])
			y2 = int(z[1][1])
			person = frame[y1:y2, x1:x2, :]
			identify_name, score = compare.run(person, origin_f, origin_name)
			# c1 = 118
			# #connect to database
			# mydb = db_connection()
			# cur = mydb.cursor()
			# info = "insert into REID(person,ctime, camera) values ('{0}',{1},{2})".format(identify_name, int(time.time()), c1)
			# cur.execute(info)
			# mydb.close()
			# time.sleep(1)
			
			if(identify_name in [ "MJ1","MJ2","MJ3","MJ4","MJ5","MJ6","MJ7","MJ8","MJ9","MJ10","MJ11","MJ12","MJ13","MJ14","MJ15","MJ16","MJ17"]):				
				identify_name = "Person_1"
				c1 = 106
				#connect to database
				if int(time.time()) - tmp > 15:
					mydb = db_connection()
					cur = mydb.cursor()
					info = "insert into REID(person,ctime, camera) values ('{0}',{1},{2})".format(identify_name, int(time.time()), c1)
					tmp = int(time.time())
					cur.execute(info)
					print("insert one record", tmp)
					mydb.close()
			elif(identify_name in ["QY1", "QY2", "QY3", "QY4", "QY5","QY6","QY7","QY8","QY9","QY10", "QY11", "QY12"]):
				identify_name = "Person_2"
				c1 = 106
				if int(time.time()) - tmp2 > 15:
					mydb = db_connection()
					cur = mydb.cursor()
					info = "insert into REID(person,ctime, camera) values ('{0}',{1},{2})".format(identify_name, int(time.time()), c1)
					tmp2 = int(time.time())
					cur.execute(info)
					print("insert one record", tmp)
					mydb.close()
			print("identify name:{}, score:{}".format(identify_name, round(1-score, 2)))
			
			#generate a new tracker for the person
			z = np.expand_dims([n for a in z for n in a], axis=0).T
			tmp_trk = Tracker() # Create a new tracker
			x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
			tmp_trk.x_state = x
			tmp_trk.predict_only()
			xx = tmp_trk.x_state
			xx = xx.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]
			tmp_trk.box = xx
			tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
			tmp_trk.personReID_info['personID'] = identify_name #assign the reidentified personID for the tracker
			tracker_list.append(tmp_trk)
			x_box.append(xx)
	
		# Deal with unmatched tracks       
	if len(unmatched_trks)>0:
		for trk_idx in unmatched_trks:
			tmp_trk = tracker_list[trk_idx]
			tmp_trk.no_losses += 1
			tmp_trk.predict_only()
			xx = tmp_trk.x_state
			xx = xx.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]
			tmp_trk.box =xx
			x_box[trk_idx] = xx
	
	# The list of tracks to be annotated and draw the figure
	good_tracker_list =[]
	for trk in tracker_list:
		if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
			good_tracker_list.append(trk)
			x_cv2 = trk.box
			trackerID_str="Unknown Person:"+str(trk.id)
			if trk.personReID_info['personID'] == "Unknown":
				frame= draw_box_label(frame, x_cv2, box_color=(0, 255, 128), personReID_info={'personID':trackerID_str}) # Draw the bounding boxes for unknown person
			else:
				frame= draw_box_label(frame, x_cv2, box_color=(0, 255, 255), personReID_info=trk.personReID_info) # Draw the bounding boxes for re-identified person
	#book keeping
	deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)

	for trk in deleted_tracks:
		track_id_list.append(trk.id)

	tracker_list = [x for x in tracker_list if x.no_losses<=max_age]

# 	#the original codes
# 	for bbox in detection_results:
# 		logger.info('coordinates: {} {}. '.
# 					format(bbox[0], bbox[1]))

# 		x1 = int(bbox[0][0])
# 		y1 = int(bbox[0][1])
# 		x2 = int(bbox[1][0])
# 		y2 = int(bbox[1][1])

# 		person = frame[y1:y2, x1:x2, :]

# 		identify_name, score = compare.run(person, origin_f, origin_name)

# 		if(identify_name in [ "MJ1", "MJ2", "MJ3", "MJ4", "MJ5"]):
# 				identify_name = "Person_1"
# 		elif(identify_name in ["QY1", "QY2", "QY3", "QY4", "QY5"]):
# 			identify_name = "Person_2"
# 			
# 		print("identify name:{}, score:{}".format(identify_name, round(1-score, 2)))
# 		
# 		bounding_boxs.append([(x1,y1), (x2,y2), identify_name+' '+str(round(1-score, 2))])
# 		#img = cam_detection.draw_rectangle(img, (x1,y1,x2,y2), identify_name+'  '+str(round((1-score), 2)))
		
		
	
			
# 	for obj in bounding_boxs:
# 		print(obj)
# 		cv2.putText(frame, obj[2], (obj[0][0], obj[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
# 		frame = cv2.rectangle(frame, obj[0], obj[1], (0, 255, 0), 2)
		
	return frame


def gen_frames():  # generate frame by frame from camera
	#stream detection
	#cap = open_cam_onboard(640, 480)
	#uri = "rtsp://admin:admin@192.168.1.106:554/stream2"
	
	uri = "rtsp://admin:edge1234@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0"
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
		#print ("before read:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
		if counter % 5 != 0:
			ret, frame = cap.read()
			#print ("after read", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
			continue

		#logger.info("FPS: {0:.2f}".format(frame_rate_calc))
		#cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
		#			cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

		#result = api.get_person_bbox(frame, thr=0.6)  #add functions to this line
		frame = handle_frames(frame)
		#print ("after handle", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

		t2 = time.time()
		#print("one frame takes {0:.2f}".format(t2-t1))
		frame_rate_calc = 1 / (t2 - t1)
		#if frame_rate_calc < 15:
		#	frame_rate_calc = 2*frame_rate_calc

		cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
					cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

		#if counter < 5:
		#	plt.imshow(frame[:, :, ::-1])
		#	plt.show()
		#	continue		

		# show the frame
		#cv2.imshow("Stream from EdgeNX1", frame)
		#key = cv2.waitKey(1) & 0xFF

		#t2 = cv2.getTickCount()
		
		#time1 = (t2 - t1) / freq
		#frame_rate_calc = 1 / time1
		#print("one frame takes {0:.2f}".format(t2-t1))
		
		(flag, outputFrame) = cv2.imencode(".jpg", frame)
		yield (b'--frame\r\n'
					   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(outputFrame) + b'\r\n')
		

		# if the `q` key was pressed, break from the loop
		#if key == ord("q"):
		#	break
	
@app.route('/video_feed')
def video_feed():
	#Video streaming route. Put this in the src attribute of an img tag
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
	"""Video streaming home page."""
	return render_template('index.html')

if __name__ == '__main__':
	img = cv2.imread('example.jpg')
	img = handle_frames(img)
	#plt.imshow(img[:, :, ::-1])
	print("show frame")
	#plt.show()
	app.run(host='0.0.0.0', port='5000')
	#gen_frames()

