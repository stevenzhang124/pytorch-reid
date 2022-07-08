#!/usr/bin/env python3
from threadpoolctl import threadpool_limits
from flask import Flask, render_template, Response
from pathlib import Path
from types import SimpleNamespace
import argparse
import logging
import time
import json
import cv2

import utils.fastmot as fastmot
import utils.fastmot.models 
from utils import ConfigDecoder, Profiler

# import fastmot
# import fastmot.models
# from fastmot.utils import ConfigDecoder, Profiler

# import pycuda.driver as cuda
# import pycuda.autoinit
import sys, select, os
import keyboard

import mysql.connector

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
group = parser.add_mutually_exclusive_group()
required.add_argument('-i', '--input_uri', metavar="URI", required=True, help=
                    'URI to input stream\n'
                    '1) video file (e.g. input.mp4)\n'
                    '2) MIPI CSI camera (e.g. csi://0)\n'
                    '3) USB or V4L2 camera (e.g. /dev/video0)\n'
                    '4) RTSP stream (rtsp://<user>:<password>@<ip>:<port>)\n'
                    '5) Edge RTSP stream (edge://)')
required.add_argument('-c','--camera', help='Input camera number: (e.g. input 106 for 114)'
    ' 106(114)\n113(105)\n115(102)\n116(107)\n117(103)\n118(109)')
required.add_argument('-s','--num_cams', help='Number of Cameras', type=int)
optional.add_argument('-con', '--config', metavar="FILE",
                          default=Path(__file__).parent / 'cfg' / 'mot.json',
                          help='path to JSON configuration file')
optional.add_argument('-m', '--mot', action='store_true', help='run multiple object tracker')
optional.add_argument('-l', '--labels', metavar="FILE",
                        help='path to label names (e.g. coco.names)')
optional.add_argument('-o', '--output_uri', metavar="URI",
                    help='URI to output stream (e.g. output.mp4)')
optional.add_argument('-t', '--txt', metavar="FILE",
                        help='output MOT Challenge txt results (e.g. eval/results/MOT20-01.txt)')
optional.add_argument('-g', '--gui', action='store_true', help='enable display')
optional.add_argument('-x', '--exclusive', action='store_true', help='bool for http display')
optional.add_argument('-d', '--database', action='store_true', help='bool for using database')
optional.add_argument('-r', '--par', action='store_true', help='bool for using attribute recognition')
group.add_argument('-q', '--quiet', action='store_true', help='reduce output verbosity')
group.add_argument('-v', '--verbose', action='store_true', help='verbose output for debugging')

args = parser.parse_args()

app = Flask(__name__)

def db_connection(config):
    try:
        mydb = mysql.connector.connect(
            host=config.host,
            user=config.user,
            port=config.port,
            database=config.database,
            passwd=config.passwd,
            autocommit=config.autocommit)
        return mydb
    except mysql.connector.Error as err:
      if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
      elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
      else:
        print(err)
    else:
        return mydb

def app_main():
    # set up logging
    # logging.basicConfig(filename='app2.log', level=logging.DEBUG, format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # logger = logging.getLogger(fastmot.__name__)
    # logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(fastmot.__name__)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # # load config file
    # with open(Path(__file__).parent / 'cfg' / 'mot.json') as config_file:
    #     config = json.load(config_file, cls=fastmot.utils.ConfigDecoder)

    # load config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))

    # load labels if given
    if args.labels is not None:
        with open(args.labels) as label_file:
            label_map = label_file.read().splitlines()
            fastmot.models.set_label_map(label_map)
    
    elapsed_time = 0
    # stream = fastmot.VideoIO(config['size'], config['video_io'], args.input_uri, args.camera, args.output_uri)
    stream = fastmot.VideoIO(config.resize_to, args.input_uri, args.camera, args.output_uri, **vars(config.stream_cfg))
    
    db = fastmot.Database(args.camera, db_connection(config.database), args.num_cams) if (args.database or args.par) else None

    mot = None
    txt = None
    if args.mot:
        # draw = args.gui or args.output_uri is not None
        draw = args.gui or "edge" in args.input_uri
        draw = True
        # mot = fastmot.MOT(config['size'], stream.capture_dt, config['mot'], args.camera, db, stream.fps,
        #                   draw=draw, verbose=args.verbose, cuda_ctx=pycuda.autoinit.context)
        mot = fastmot.MOT(config.resize_to, args.camera, db, stream.cap_fps, args.par, **vars(config.mot_cfg),
         draw=draw, verbose=args.verbose)# cuda_ctx=pycuda.autoinit.context)
        mot.reset(stream.cap_dt)

        if args.txt is not None:
            assert Path(args.txt).suffix == '.txt'
            Path(args.txt).parent.mkdir(parents=True, exist_ok=True)
            txt = open(args.txt, 'w')
    if args.gui:
        cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
    
    if args.camera != "113":
        print("Wait starts...")
        time.sleep(5)
        print("Wait over...")

    logger.info('Starting video capture...')
    stream.start_capture()
    try:
        tic = time.perf_counter()
        init_time = time.time()
        with Profiler('app') as prof:
            while not args.gui or cv2.getWindowProperty("Video", 0) >= 0:
                frame = stream.read()
                if frame is None:
                    break

                if args.mot:
                    mot.step(frame)
                    # avg_fps = min(stream.cap_fps,1 / (time.perf_counter() - tic))
                    # mot.getAverages(avg_fps, logger)
                    if txt is not None:
                        for track in mot.visible_tracks():
                            # MOT17 dataset is usually of size 1920x1080, modify this otherwise
                            # orig_size = (640, 480)
                            # tl = track.tlbr[:2] / config['size'] * orig_size
                            # br = track.tlbr[2:] / config['size'] * orig_size
                            # w, h = br - tl + 1
                            # txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                            #           f'{w:.6f},{h:.6f},-1,-1,-1\n')
                            tl = track.tlbr[:2] / config.resize_to * stream.resolution
                            br = track.tlbr[2:] / config.resize_to * stream.resolution
                            w, h = br - tl + 1
                            txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                                      f'{w:.6f},{h:.6f},-1,-1,-1\n')
                if args.gui:
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if args.output_uri is not None:
                    stream.write(frame)


                if not args.exclusive:
                    (flag, outputFrame) = cv2.imencode(".jpg", frame)
                    yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(outputFrame) + b'\r\n')
                
                # if args.mot and time.time() - init_time >= 1 * 60 :
                #     mot.getAverages(time.perf_counter() - tic,logger)
                #     init_time = time.time()

        toc = time.perf_counter()
        elapsed_time = toc - tic
            
    finally:
        # clean up resources
        if txt is not None:
            txt.close()
        stream.release()
        cv2.destroyAllWindows()

    # Final Times
    avg_fps = round(mot.frame_count / prof.duration)
    # mot.tracker.getTimes()
    mot.getAverages(avg_fps, logger)
    return
    


# def write_log(duration):
#     file_name = 'duration.txt'
#     with open(file_name,'a+') as file:
#         file.writelines(str(duration) + '\n')

#     return 0
    

def main():
    # set up logging
    # logging.basicConfig(filename='app2.log',filemode='w',format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # logger = logging.getLogger(fastmot.__name__)
    # logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(fastmot.__name__)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # # load config file
    # with open(Path(__file__).parent / 'cfg' / 'mot.json') as config_file:
    #     config = json.load(config_file, cls=fastmot.utils.ConfigDecoder)

    # load config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))

    # load labels if given
    if args.labels is not None:
        with open(args.labels) as label_file:
            label_map = label_file.read().splitlines()
            fastmot.models.set_label_map(label_map)
    
    elapsed_time = 0
    # stream = fastmot.VideoIO(config['size'], config['video_io'], args.input_uri, args.camera, args.output_uri)
    stream = fastmot.VideoIO(config.resize_to, args.input_uri, args.camera, args.output_uri, **vars(config.stream_cfg))
    
    db = fastmot.Database(args.camera, db_connection(config.database), args.num_cams) if (args.database or args.par) else None

    mot = None
    txt = None
    if args.mot:
        # draw = args.gui or args.output_uri is not None
        draw = args.gui or "edge" in args.input_uri
        # mot = fastmot.MOT(config['size'], stream.capture_dt, config['mot'], args.camera, db, stream.fps,
        #                   draw=draw, verbose=args.verbose, cuda_ctx=pycuda.autoinit.context)
        mot = fastmot.MOT(config.resize_to, args.camera, db, stream.cap_fps, args.par, **vars(config.mot_cfg),
         draw=draw, verbose=args.verbose)# cuda_ctx=pycuda.autoinit.context)
        mot.reset(stream.cap_dt)

        if args.txt is not None:
            assert Path(args.txt).suffix == '.txt'
            Path(args.txt).parent.mkdir(parents=True, exist_ok=True)
            txt = open(args.txt, 'w')
    if args.gui:
        cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
    
    if args.camera != "115":
        print("Wait starts...")
        time.sleep(5)
        print("Wait over...")

    logger.info('Starting video capture...')
    stream.start_capture()
    try:
        tic = time.perf_counter()
        init_time = time.time()
        with Profiler('app') as prof:
            while not args.gui or cv2.getWindowProperty("Video", 0) >= 0:
                frame = stream.read()
                # frame_time = stream.get_frame_time()
                if frame is None:
                    break

                if args.mot:
                    # t1 = time.time()
                    # mot.step(frame, frmae_time)
                    mot.step(frame)
                    # t2 = time.time()
                    # write_log(t2-t1)
                    # avg_fps = min(stream.cap_fps,1 / (time.perf_counter() - tic))
                    # write_log(avg_fps)
                    # mot.getAverages(avg_fps, logger)
                    if txt is not None:
                        for track in mot.visible_tracks():
                            # MOT17 dataset is usually of size 1920x1080, modify this otherwise
                            # orig_size = (640, 480)
                            # tl = track.tlbr[:2] / config['size'] * orig_size
                            # br = track.tlbr[2:] / config['size'] * orig_size
                            # w, h = br - tl + 1
                            # txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                            #           f'{w:.6f},{h:.6f},-1,-1,-1\n')
                            tl = track.tlbr[:2] / config.resize_to * stream.resolution
                            br = track.tlbr[2:] / config.resize_to * stream.resolution
                            w, h = br - tl + 1
                            txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                                      f'{w:.6f},{h:.6f},-1,-1,-1\n')
                if args.gui:
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if args.output_uri is not None:
                    stream.write(frame)

                # if not args.exclusive:
                #     (flag, outputFrame) = cv2.imencode(".jpg", frame)
                #     yield (b'--frame\r\n'
                #                b'Content-Type: image/jpeg\r\n\r\n' + bytearray(outputFrame) + b'\r\n')

                # if args.mot and time.time() - init_time >= 1 * 60 :
                #     mot.getAverages(time.perf_counter() - tic,logger)
                #     init_time = time.time()

        toc = time.perf_counter()
        elapsed_time = toc - tic
            
    finally:
        # clean up resources
        if txt is not None:
            txt.close()
        stream.release()
        cv2.destroyAllWindows()

    # Final Times
    avg_fps = round(mot.frame_count / prof.duration)
    # mot.tracker.getTimes()
    mot.getAverages(avg_fps, logger)


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(app_main(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        with open(args.config) as cfg_file:
            config = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))
        init_db = fastmot.Database(args.camera, db_connection(config.database), args.num_cams)
        init_db.init()
        init_db.database.commit()
        init_db.database.close()
        if not args.exclusive:
            app.run(host='0.0.0.0', port='5000')
        else:
            main()
            # app.run(host='192.168.1.103',port='5000')
