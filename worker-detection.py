import requests
from flask import Flask, request
import cv2
from pedestrian_detection_ssdlite import api
import sys
from concurrent.futures import ThreadPoolExecutor
import time
# from reid import cam_reid

app = Flask(__name__)
executor = ThreadPoolExecutor(3)


# reid_mode = cam_reid.reid_model()
# # encode origin image
# compare = cam_reid.Compare(model=reid_mode, origin_img="./image/origin")
# origin_f, origin_name = compare.encode_origin_image()

@app.route('/detection', methods=['POST'])
def detection():
	t1 = time.time()
	data = request.form.to_dict()
	img = request.files["file"]
	img.save('a.jpg')
	image = cv2.imread('a.jpg')
	
	detection_test = api.get_person_bbox(image, thr=0.5)
	print(len(detection_test))
	t2 = time.time()
	print('detection consumes', t2-t1)
	#time.sleep(2)

	url = data['offload-url']
	#count = 0
	
	executor.submit(offload, url, detection_test, image)
	# 	count=0
	# count += 1
	return str(t2-t1)

def offload(url, detection_test, img):
	identify_num = 0
	persons = {}
	for bbox in detection_test:
		x1 = int(bbox[0][0])
		y1 = int(bbox[0][1])
		x2 = int(bbox[1][0])
		y2 = int(bbox[1][1])
		
		person_string = str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2)
		persons[str(identify_num)] = person_string
		identify_num += 1

	print(persons)
	file = {"file": ("file_name.jpg", cv2.imencode(".jpg", img)[1].tobytes(), "image/jpg")}
	url = 'http://' + url + ':5002/reid'
	t1 = time.time()
	info = requests.post(url, data=persons, files=file)
	t2 = time.time()
	print('reid consumes', t2-t1)


@app.route('/kill-detection', methods=['GET'])
def kill():
	sys.exit()



if __name__ == '__main__':
	img = cv2.imread('example.jpg')
	detection = api.get_person_bbox(img, thr=0.5)
	app.run(host='0.0.0.0', port=5001, debug=False, threaded = True)
	# i = 0
	# t_total = 0
	# while(i < 10 ):
	# 	file = 'edge' + str(i+1) + '.jpg'
	# 	img = cv2.imread('./test_img/' +file)
	# 	t1 = time.time()
	# 	detection = api.get_person_bbox(img, thr=0.5)
	# 	print(detection)
	# 	t2 = time.time()
	# 	t_tmp = t2-t1
	# 	t_total = t_total + t_tmp
	# 	print('detection' + file + 'consumes', t_tmp)
	# 	i += 1
	# print("average is", t_total/10)

