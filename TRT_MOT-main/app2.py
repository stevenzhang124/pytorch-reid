#!/usr/bin/env python3
from flask import Flask, render_template, Response
from pathlib import Path
import argparse
import logging
import time
import json
import cv2

import fastmot

# import pycuda.driver as cuda
import pycuda.autoinit
import sys, select, os
import keyboard

import mysql.connector
import time


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-m', '--mot', action='store_true', help='run multiple object tracker')
parser.add_argument('-i', '--input_uri', metavar="URI", required=True, help=
                    'URI to input stream\n'
                    '1) video file (e.g. input.mp4)\n'
                    '2) MIPI CSI camera (e.g. csi://0)\n'
                    '3) USB or V4L2 camera (e.g. /dev/video0)\n'
                    '4) RTSP stream (rtsp://<user>:<password>@<ip>:<port>)\n'
                    '5) Edge RTSP stream (edge://)')
parser.add_argument('-o', '--output_uri', metavar="URI",
                    help='URI to output stream (e.g. output.mp4)')
parser.add_argument('-l', '--log', metavar="FILE",
                    help='output a MOT Challenge format log (e.g. eval/results/mot17-04.txt)')
parser.add_argument('-g', '--gui', action='store_true', help='enable display')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose output for debugging')
parser.add_argument('-c','--camera', help='Input camera number: (e.g. input 106 for 114)'
    ' 106(114)\n113(105)\n115(102)\n116(107)\n117(103)\n118(109)', required=True)
parser.add_argument('-s','--num_cams', help='Number of Cameras', type=int, required=True)
args = parser.parse_args()

app = Flask(__name__)

def db_connection(config):
        mydb = mysql.connector.connect(
            host=config["host"],
            user=config["user"],
            port=config["port"],
            database=config["database"],
            passwd=config["passwd"],
            autocommit=config["autocommit"])
        return mydb

def main():
    # set up logging
    logging.basicConfig(format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(fastmot.__name__)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # load config file
    with open(Path(__file__).parent / 'cfg' / 'mot.json') as config_file:
        config = json.load(config_file, cls=fastmot.utils.ConfigDecoder)

    mot = None
    log = None
    elapsed_time = 0
    stream = fastmot.VideoIO(config['size'], config['video_io'], args.input_uri, args.camera, args.output_uri)
    db = fastmot.Database(args.camera,db_connection(config["database"]), args.num_cams)

    if args.mot:
        # draw = args.gui or args.output_uri is not None
        draw = True
        mot = fastmot.MOT(config['size'], stream.capture_dt, config['mot'], args.camera, db, stream.fps,
                          draw=draw, verbose=args.verbose, cuda_ctx=pycuda.autoinit.context)
        if args.log is not None:
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            log = open(args.log, 'w')
    if args.gui:
        cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
    
    if args.camera != "113":
        print("Wait starts...")
        time.sleep(5)
        print("Wait over...")

    logger.info('Starting video capture...')
    # print("Stream is",stream)
    stream.start_capture()
    try:
        tic = time.perf_counter()
        init_time = time.time()
        while not args.gui or cv2.getWindowProperty("Video", 0) >= 0:
            frame = stream.read()
            if frame is None:
                break

            if args.mot:
                mot.step(frame)
                if log is not None:
                    for track in mot.visible_tracks:
                        # MOT17 dataset is usually of size 1920x1080, modify this otherwise
                        orig_size = (640, 480)
                        tl = track.tlbr[:2] / config['size'] * orig_size
                        br = track.tlbr[2:] / config['size'] * orig_size
                        w, h = br - tl + 1
                        log.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                                  f'{w:.6f},{h:.6f},-1,-1,-1\n')
            if args.gui:
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if args.output_uri is not None:
                stream.write(frame)

            (flag, outputFrame) = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + bytearray(outputFrame) + b'\r\n')
            
            if args.mot and time.time() - init_time >= 1 * 60 :
                mot.getAverages(time.perf_counter() - tic,logger)
                init_time = time.time()

        toc = time.perf_counter()
        elapsed_time = toc - tic
            
    finally:
        # clean up resources
        if log is not None:
            log.close()
        stream.release()
        cv2.destroyAllWindows()

    # Final Times
    mot.getAverages(elapsed_time,logger)
        

# @app.route('/')
# def index():
#     """Video streaming home page."""
#     return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    with open(Path(__file__).parent / 'cfg' / 'mot.json') as config_file:
        config = json.load(config_file, cls=fastmot.utils.ConfigDecoder)
    init_db = fastmot.Database(args.camera,db_connection(config["database"]), args.num_cams)
    init_db.init()
    init_db.database.commit()
    init_db.database.close()
    app.run(host='0.0.0.0', port='5000')
    # app.run(host='192.168.1.103',port='5000')
