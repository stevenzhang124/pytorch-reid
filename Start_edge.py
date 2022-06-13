import flask
import subprocess

app = flask.Flask(__name__)
#curl -v -d "nodeName=edge" http://127.0.0.1:5000/listen
@app.route("/listen", methods=["POST"])
def listen():
	Node = flask.request.form.get('nodeName')
	#cmd1 = "cd /home/edge/Documents/pedestrian-detection-ssdlite && python3 test3.py"
	cmd1 = "cd /home/edge/Documents/pytorch-reid && python3 test10.py"
	if Node == 'edge':
		subprocess.Popen(cmd1, shell=True)

	OK = "start application\n"
	return OK

if __name__ == "__main__":
	app.run(host='0.0.0.0', port = 5001)