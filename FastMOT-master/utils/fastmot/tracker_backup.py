from types import SimpleNamespace
from collections import OrderedDict, deque
import itertools
import logging
import numpy as np
import numba as nb

from ..track import Track
from ..flow import Flow
from ..kalman_filter import MeasType, KalmanFilter
from .attribute_recognizer import AttributeRecognizer
from ..distance import Metric, cdist, iou_dist
from ..matching import linear_assignment, greedy_match, fuse_motion, gate_cost
from ..rect import as_tlbr, to_tlbr, ios, bbox_ious, find_occluded, area, crop, get_center
from .. import Profiler
import cv2



LOGGER = logging.getLogger(__name__)


class MultiTracker:
    def __init__(self, size, metric, camera, database, par,
                 buffer_size=30,
                 max_age=6,
                 age_penalty=2,
                 motion_weight=0.2,
                 max_assoc_cost=0.9,
                 max_reid_cost=0.45,
                 iou_thresh=0.4,
                 duplicate_thresh=0.8,
                 occlusion_thresh=0.7,
                 conf_thresh=0.5,
                 confirm_hits=1,
                 history_size=50,
                 kalman_filter_cfg=None,
                 flow_cfg=None):
        """Class that uses KLT and Kalman filter to track multiple objects and
        associates detections to tracklets based on motion and appearance.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        metric : {'euclidean', 'cosine'}
            Feature distance metric to associate tracks.
        max_age : int, optional
            Max number of undetected frames allowed before a track is terminated.
            Note that skipped frames are not included.
        age_penalty : int, optional
            Scale factor to penalize KLT measurements for tracks with large age.
        motion_weight : float, optional
            Weight for motion term in matching cost function.
        max_assoc_cost : float, optional
            Max matching cost for valid primary association.
        max_reid_cost : float, optional
            Max ReID feature dissimilarity for valid reidentification.
        iou_thresh : float, optional
            IoU threshold for association with unconfirmed and unmatched active tracks.
        duplicate_thresh : float, optional
            Track overlap threshold for removing duplicate tracks.
        occlusion_thresh : float, optional
            Detection overlap threshold for nullifying the extracted embeddings for association/reID.
        conf_thresh : float, optional
            Detection confidence threshold for starting a new track.
        confirm_hits : int, optional
            Min number of detections to confirm a track.
        history_size : int, optional
            Max size of track history to keep for reID.
        kalman_filter_cfg : SimpleNamespace, optional
            Kalman Filter configuration.
        flow_cfg : SimpleNamespace, optional
            Flow configuration.
        """
        self.size = size
        self.metric = Metric[metric.upper()]
        assert max_age >= 1
        self.max_age = max_age
        assert age_penalty >= 1
        self.age_penalty = age_penalty
        assert 0 <= motion_weight <= 1
        self.motion_weight = motion_weight
        assert 0 <= max_assoc_cost <= 2
        self.max_assoc_cost = max_assoc_cost
        assert 0 <= max_reid_cost <= 2
        self.max_reid_cost = max_reid_cost
        assert 0 <= iou_thresh <= 1
        self.iou_thresh = iou_thresh
        assert 0 <= duplicate_thresh <= 1
        self.duplicate_thresh = duplicate_thresh
        assert 0 <= occlusion_thresh <= 1
        self.occlusion_thresh = occlusion_thresh
        assert 0 <= conf_thresh <= 1
        self.conf_thresh = conf_thresh
        assert confirm_hits >= 1
        self.confirm_hits = confirm_hits
        assert history_size >= 0
        self.history_size = history_size

        if kalman_filter_cfg is None:
            kalman_filter_cfg = SimpleNamespace()
        if flow_cfg is None:
            flow_cfg = SimpleNamespace()

        self.database = database
        self.camera = camera
        self.cameraids = {"106":1, "113":2, "115":3, "116":4, "117":5, "118":6}

        self.next_id = max(1, self.database.get_nextid()) if database else 1 
        self.tracks = {}
        self.lost = OrderedDict()
        self.hist_tracks = OrderedDict()
        self.db_tracks = {}
        self.count_tracks = {}
        self.in_tracks, self.out_tracks = {}, {}
        self.duplicates = []
        self.kf = KalmanFilter(**vars(kalman_filter_cfg))
        self.flow = Flow(self.size, **vars(flow_cfg))
        self.frame_rect = to_tlbr((0, 0, *self.size))

        self.klt_bboxes = {}
        self.homography = None
        self.buffer_size = buffer_size
        if par:
            LOGGER.info('Loading attribute recognition model...')
        self.par = AttributeRecognizer() if par else None
        self.entries = max(0, self.database.get_entry()) if database else 0
        self.exits = max(0, self.database.get_exit()) if database else 0

    def reset(self, dt):
        """Reset the tracker for new input context.

        Parameters
        ----------
        dt : float
            Time interval in seconds between each frame.
        """
        self.kf.reset_dt(dt)
        self.lost.clear()
        Track._count = 0

    def init(self, frame, detections):
        """Initializes the tracker from detections in the first frame.

        Parameters
        ----------
        frame : ndarray
            Initial frame.
        detections : recarray[DET_DTYPE]
            Record array of N detections.
        """
        self.next_id = self.database.get_nextid() if self.database else self.next_id
        self.tracks.clear()
        self._update_tracks()
        self.flow.init(frame)
        for det in detections:
            state = self.kf.create(det.tlbr)
            new_trk = Track(frame_id=0, trk_id=self.next_id, tlbr=det.tlbr, state=state, 
                label=det.label, confirm_hits=self.confirm_hits)
            self.tracks[new_trk.trk_id] = new_trk
            if self.database:
                    self.database.update_nextid(self.next_id)
            self.next_id = max(self.next_id + 1, self.database.get_nextid())  if self.database else self.next_id + 1
            LOGGER.debug(f"{'Detected(Init):':<14}{new_trk}")
        self.entries = self.database.get_entry() if self.database else self.entries
        self.exits = self.database.get_exit() if self.database else self.exits

    def track(self, frame, frame_time):
        """Convenience function that combines `compute_flow` and `apply_kalman`.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        self.compute_flow(frame)
        self.apply_kalman(frame, frame_time)

    def compute_flow(self, frame):
        """Computes optical flow to estimate tracklet positions and camera motion.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        active_tracks = [track for track in self.tracks.values() if track.active]
        self.klt_bboxes, self.homography = self.flow.predict(frame, active_tracks)
        # if self.homography is None:
        #     # clear tracks when camera motion cannot be estimated
        #     self.tracks.clear()

    def apply_kalman(self, frame, frame_time):
        """Performs kalman filter predict and update from KLT measurements.
        The function should be called after `compute_flow`.
        """
        for trk_id, track in list(self.tracks.items()):
            mean, cov = track.state
            # mean, cov = self.kf.warp(mean, cov, self.homography)
            mean, cov = self.kf.predict(mean, cov)
            if trk_id in self.klt_bboxes:
                klt_tlbr = self.klt_bboxes[trk_id]
                # give large KLT uncertainty for occluded tracks
                # usually these with large age and low inlier ratio
                std_multiplier = max(self.age_penalty * track.age, 1) / track.inlier_ratio
                mean, cov = self.kf.update(mean, cov, klt_tlbr, MeasType.FLOW, std_multiplier)
            next_tlbr = as_tlbr(mean[:4])
            track.update(next_tlbr, (mean, cov))

            # Person Count
            x, y = get_center(next_tlbr)
            if self.camera == "113":
                cv2.line(frame, (150, 0), (150, self.size[1]), (0, 255, 0), thickness=2)
            elif self.camera == "106":
                cv2.line(frame, (0, 150), (self.size[0], 150), (0, 255, 0), thickness=2)
            if track.confirmed :
                if self.camera == "113" and (self.in_tracks[track.trk_id] == -1 or self.out_tracks[track.trk_id] == -1):
                    # direction = self.doorThreshold(x, y, track, '113')
                    # # direction = direction if direction != self.count_tracks[track.trk_id] and direction != -1 else self.count_tracks[track.trk_id]
                    # direction = direction if direction != self.count_tracks[track.trk_id] else self.count_tracks[track.trk_id]
                    # self.movementDirection(direction, "A2", track, frame_time)
                    # self.count_tracks[track.trk_id] = direction
                    
                    direction = self.doorThreshold(x, y, track, '113')
                    if direction == 1 and self.in_tracks[track.trk_id] == -1 :
                        self.in_tracks[track.trk_id] = direction
                    elif direction == 0 and self.out_tracks[track.trk_id] == -1 :
                        self.out_tracks[track.trk_id] = direction
                    else:
                        direction = -1
                    self.movementDirection(direction, "A2", track, frame_time)

                elif self.camera == "106" and (self.in_tracks[track.trk_id] == -1 or self.out_tracks[track.trk_id] == -1):
                    # direction = self.doorThreshold(x, y, track, '106')
                    # # direction = direction if direction != self.count_tracks[track.trk_id] and direction != -1 else self.count_tracks[track.trk_id]
                    # direction = direction if direction != self.count_tracks[track.trk_id] else self.count_tracks[track.trk_id]
                    # self.movementDirection(direction, "B2", track, frame_time)
                    # self.count_tracks[track.trk_id] = direction

                    direction = self.doorThreshold(x, y, track, '106')
                    if direction == 1 and self.in_tracks[track.trk_id] == -1 :
                        self.in_tracks[track.trk_id] = direction
                    elif direction == 0 and self.out_tracks[track.trk_id] == -1 :
                        self.out_tracks[track.trk_id] = direction
                    else:
                        direction = -1
                    self.movementDirection(direction, "B2", track, frame_time)

                elif self.camera == "115":
                    if self.doorThreshold(x, y, track, '115'):
                        direction = self.getDirection(track)
                        self.movementDirection(direction, "C2", track, frame_time)
            if ios(next_tlbr, self.frame_rect) < 0.5:
                
                    # self._mark_lost(trk_id)
                    # LOGGER.info(f"{'     -->:':<14}{ios(next_tlbr,self.frame_rect)}")
                    # if self.database and track.avg_feat.is_valid:
                    #     #attr_data = self.par.predict(crop(frame,track.tlbr))
                    #     attr_data = None, None, None, None, None 
                    #     self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3], attr_data)
                    #     #print("Track is {} ID is {}".format(track, trk_id))
                LOGGER.info(f"{'Lost(Out):':<14}{track, ios(next_tlbr,self.frame_rect)}")
                self._mark_lost(trk_id)
                # else:
                #     del self.tracks[trk_id]

    def update(self, frame, frame_id, detections, embeddings):
        """Associates detections to tracklets based on motion and feature embeddings.

        Parameters
        ----------
        frame_id : int
            The next frame ID.
        detections : recarray[DET_DTYPE]
            Record array of N detections.
        embeddings : ndarray
            NxM matrix of N extracted embeddings with dimension M.
        """
        with Profiler('preproc'):   
            #print("###################################--{}--########################################".format(frame_id))

            self.next_id = max(self.next_id, self.database.get_nextid())  if self.database else self.next_id

            occluded_det_mask = find_occluded(detections.tlbr, self.occlusion_thresh)
            confirmed_by_depth, unconfirmed = self._group_tracks_by_depth()
            u_det_ids = list(range(len(detections)))
            # LOGGER.info(f"{'Detections:':<14}{u_det_ids}")
            # LOGGER.info(f"{'Occluded:':<14}{occluded_det_mask}")
            # LOGGER.info(f"{'Confirmed:':<14}{confirmed_by_depth}")
            # LOGGER.info(f"{'Unconfirmed:':<14}{unconfirmed}")
            # LOGGER.info(f"{'Lost:':<14}{self.lost.keys()}")
            # LOGGER.info(f"{'DB:':<14}{self.db_tracks.keys()}")

            # #print("DETECTIONS=",u_det_ids)
            # #print("OCCLUDED =",occluded_det_mask)
            # #print("CONFIRMED =", confirmed_by_depth)
            # #print("UNCONFIRMED =", unconfirmed)
            # #print("LOST =", self.lost.keys())

            # association with motion and embeddings, tracks with small age are prioritized
            matches1 = []
            u_trk_ids1 = []

        with Profiler('m1'):
            for depth, trk_ids in enumerate(confirmed_by_depth):
                # #print("depth = {} , trk_ids = {} ".format(depth,trk_ids))
                if len(u_det_ids) == 0:
                    u_trk_ids1.extend(itertools.chain.from_iterable(confirmed_by_depth[depth:]))
                    break
                if len(trk_ids) == 0:
                    continue
                u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
                u_occluded_dmask = occluded_det_mask[u_det_ids]
                cost = self._matching_cost(trk_ids, u_detections, u_embeddings, u_occluded_dmask)
                matches, u_trk_ids, u_det_ids = linear_assignment(cost, trk_ids, u_det_ids, 1, "Matches1")
                matches1 += matches
                u_trk_ids1 += u_trk_ids

        with Profiler('m2'):   
            # 2nd association with IoU
            active = [trk_id for trk_id in u_trk_ids1 if self.tracks[trk_id].active]
            u_trk_ids1 = [trk_id for trk_id in u_trk_ids1 if not self.tracks[trk_id].active]
            u_detections = detections[u_det_ids]
            cost = self._iou_cost(active, u_detections)
            matches2, u_trk_ids2, u_det_ids = linear_assignment(cost, active, u_det_ids, 1, "Matches2")


        with Profiler('m3'):   
            # 3rd association with unconfirmed tracks
            u_detections = detections[u_det_ids]
            cost = self._iou_cost(unconfirmed, u_detections)
            matches3, u_trk_ids3, u_det_ids = linear_assignment(cost, unconfirmed, u_det_ids, 1, "Matches3")

            matches = itertools.chain(matches1, matches2, matches3)
            u_trk_ids = itertools.chain(u_trk_ids1, u_trk_ids2, u_trk_ids3)

        with Profiler('mrect'):   
            # rectify matches that may cause duplicate tracks
            matches, u_trk_ids = self._rectify_matches(matches, u_trk_ids, detections)
            updated_found = []


        with Profiler('rfound'):   
            # update matched tracks
            for trk_id, det_id in matches:
                track = self.tracks[trk_id]
                det = detections[det_id]
                mean, cov = self.kf.update(*track.state, det.tlbr, MeasType.DETECTOR)
                next_tlbr = as_tlbr(mean[:4])
                is_valid = not occluded_det_mask[det_id]
                track.add_detection(frame_id, next_tlbr, (mean, cov), embeddings[det_id], is_valid)
                if track.hits >= self.confirm_hits:
                    min_hits_confirmation = 8
                    if track.hits == self.confirm_hits:
                        x, y = get_center(track.tlbr)
                        # if self.camera == "113":
                        #     # if (x <= 220 and y <= 170):
                        #     if self.doorThreshold(x, y, track, '113'):
                        #         self.movementDirection(0, "A", track)
                        # elif self.camera == "106":
                        #     if self.doorThreshold(x, y, track, '106'):
                        #         self.movementDirection(0, "B", track)
                        # elif self.camera == "115":
                        #     if self.doorThreshold(x, y, track, '115'):
                        #         self.movementDirection(1, "C", track)
                        LOGGER.info(f"{'Found:':<14}{track}")
                        updated_found.append((trk_id, det_id))
                        # self.count_tracks[trk_id] = -1
                        self.in_tracks[trk_id] = -1
                        self.out_tracks[trk_id] = -1
                        # if self.database:
                        #     attr_data = self.par.predict(crop(frame,track.tlbr)) if self.par else None, None, None, None, None 
                        #     self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3], attr_data)
                
                if track.hits >= 10 and self.database:
                    if track.hits == 25:
                        # print("HEREEEE !!!!!!!!!!!!!!~~~~~~~~~~~~~*****&&&&&&&&&&&&&&")
                        pass
                        attr_data = self.par.predict(crop(frame,track.tlbr)) if self.par else None, None, None, None, None 
                        self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3], attr_data)
                        self.database.insert_track(trk_id, track._to_dict())
                    elif track.hits % 10 == 0:
                        pass
                        self.database.update_track(track.trk_id, track.trk_id, track._to_dict())
                if ios(next_tlbr, self.frame_rect) < 0.5:
                    is_valid = False
                    if track.confirmed:
                        LOGGER.info(f"{'Out:':<14}{track}")
                    self._mark_lost(trk_id)

        with Profiler('clean'):   
            # clean up lost tracks
            for trk_id in u_trk_ids:
                track = self.tracks[trk_id]
                track.mark_missed()
                if not track.confirmed:
                    # LOGGER.debug(f"{'Unconfirmed:':<14}{track}")
                    del self.tracks[trk_id]
                    continue
                x, y = get_center(track.tlbr)
                if track.age > self.max_age:
                    # LOGGER.info(f"{'Lost(Age):':<14}{track}")
                    if self.database:
                        self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3],(None, None, None, None, None))
                    # if self.camera == "113":
                    #     # if (x <= 220 and y <= 170):
                    #     if self.doorThreshold(x, y, track, '113'):
                    #         self.movementDirection(1,"A", track)
                    #     LOGGER.info(f"{'Lost(Age) 113:':<14}{track}")
                    # elif self.camera == "106":
                    #     if self.doorThreshold(x, y, track, '106'):
                    #         self.movementDirection(1,"B", track)
                    #     LOGGER.info(f"{'Lost(Age) 106:':<14}{track}")
                    # elif self.camera == "115":
                    #     if self.doorThreshold(x, y, track, '115'):
                    #         self.movementDirection(1,"C", track)
                    #     LOGGER.info(f"{'Lost(Age) 115:':<14}{track}")
                    self._mark_lost(trk_id)

        with Profiler('new'):   
            # start new tracks
            # u_det_ids = itertools.chain(invalid_u_det_ids, reid_u_det_ids)
            for det_id in u_det_ids:
                det = detections[det_id]
                state = self.kf.create(det.tlbr)
                new_trk = Track(frame_id=frame_id, trk_id=self.next_id, tlbr=det.tlbr,
                 state=state, label=det.label, confirm_hits=self.confirm_hits)
                self.tracks[new_trk.trk_id] = new_trk
                # LOGGER.debug(f"{'Detected:':<14}{new_trk}")
                if self.database:
                    self.database.update_nextid(self.next_id)
                self.next_id = max(self.next_id + 1, self.database.get_nextid()) if self.database else self.next_id + 1
            
    
    def box_iou2(self, rec1, rec2):
        '''
        Helper funciton to calculate the ratio between intersection and the union of
        two boxes a and b
        a[0], a[1], a[2], a[3] <->  left, top, right, bottom
        '''
        print("A{} vs B{}".format(rec1,rec2))
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1]) # H1*W1
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1]) # H2*W2
        if rec2[2] == rec2[0]:
            return 0.1

        # computing the sum_area
        sum_area = S_rec1 + S_rec2 #总面积

        # find the each edge of intersect rectangle
        left_line = max(rec1[0], rec2[0])
        right_line = min(rec1[2], rec2[2])
        top_line = max(rec1[1], rec2[1])
        bottom_line = min(rec1[3], rec2[3])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            #print("没有重合区域")
            return 0
        else:
        #print("有重合区域")
            intersect = (right_line - left_line) * (bottom_line - top_line)
            iouv=(float(intersect) / float(sum_area - intersect))*1.0

            return iouv



        # h_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
        # w_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
        # s_intsec = w_intsec * h_intsec
        # s_a = (a[2] - a[0])*(a[3] - a[1])
        # s_b = (b[2] - b[0])*(b[3] - b[1])
      
        # return float(s_intsec)/(s_a + s_b - s_intsec)


    def doorThreshold(self, x, y, track, cam='113'):
        iou_thresh = 0.2
        curr_x, curr_y = x, y
        # arr_bboxes = np.array([np.array(bbox) for bbox in track.bboxes])[-6:-1]
        # mean_bboxes = np.mean(arr_bboxes, axis=0, dtype=int)
        # prev_x, prev_y = get_center(mean_bboxes)

        if self.camera == "113":
            prev_x, prev_y = get_center(track.bboxes[0])
        elif self.camera == "106":
            prev_x, prev_y = get_center(track.bboxes[-3])
        
        
        if cam == '113':
            door1_ltrb = [0, 150, 100, 400]
            # print("Track bboxes", track, track.tlbr, track.bboxes)
            # door2_ltrb = [550, 250, 640, 450]
            # door1_tlbr = [door1_ltrb[i^1] for i in range(len(door1_ltrb))]
            # door2_tlbr = [door2_ltrb[i^1] for i in range(len(door2_ltrb))]

            # trac_bboxes = track.bboxes[0]
            # iou_1 = self.box_iou2(door1_ltrb, track.tlbr) # Though named tlbr, "self.tlbr" is in ltrb form
            # iou_2 = self.box_iou2(door2_ltrb, track.tlbr)

            # door1 = (x <= 150 and (y >= 150 and y <= 400))
            # door1 = (x <= 150) and (prev_x > 150) or (x > 150) and (prev_x <= 150)
            # if track.trk_id == 29 or track.trk_id == 25 or track.trk_id == 44:
            #     print("X,Y",x,y)
            #     print("Prev_X,Prev_Y",prev_x,prev_y)
            if (x < 150) and (prev_x >= 150): #going left
                # print("call it")
                return 1
            elif (x > 150) and (prev_x <= 150): #going right
                return 0
            else:
                return -1
            # door2 = (x >= 400 and y >= 250 and iou_2 >= 0.1)
            # LOGGER.info("LEFT IOU = {}".format(iou_1))
            # LOGGER.info("RIGHT IOU = {}".format(iou_2))
            # return door1 #or door2
        if cam == '106':
            # door1_ltrb = [0, 50, 10, 200]
            # door2_ltrb = [550, 60, 570, 125]
            # door1_ltrb = [20, 50, 90, 150]
            # iou_1 = self.box_iou2(door1_ltrb, track.tlbr)
            # # iou_2 = self.box_iou2(door2_ltrb, track.tlbr)

            # # door1 = (x <= 75) and (y <= 200 ) and (iou_1 > iou_thresh)
            # door1 = (x <= 100) and (y <= 150 ) #and (iou_1 > 0.05)
            # LOGGER.info("the iou is {}".format(iou_1))
            # if track.trk_id == 19:   
            #     print("X,Y",x,y)
            #     print("Prev_X,Prev_Y",prev_x,prev_y)
            if (y < 150) and (prev_y >= 150): #going left
                # if track.trk_id == 4:
                #     print("X,Y",x,y)
                #     print("Prev_X,Prev_Y",prev_x,prev_y)
                # print("call it")
                return 1
            elif (y > 150) and (prev_y <= 150): #going right
                return 0
            else:
                return -1
            # door2 =  (500 <= x <= 550) and (50 <= y <= 100) and (iou_2 > iou_thresh)
            # LOGGER.info("the iou is {}".format(iou_2))
            # return door1 or door2
        if cam == '115':
            door1_ltrb = [290, 30, 300, 100]
            iou_1 = self.box_iou2(door1_ltrb, track.tlbr)

            door1 = (x <= 420) and (y <= 200) and (iou_1 > iou_thresh)
            LOGGER.info("the iou is {}".format(iou_1))
            return door1 
    

    def movementDirection(self, direction, doorname, track, frame_time): # direction:  1(in), 0(out)
        if direction == 1:
            print(frame_time)
            LOGGER.info(f"{'Enter_{}:'.format(doorname):<14}{track}")
            if self.database:
                self.database.update_entry(1)
                self.entries = self.database.get_entry()
            else:
                self.entries += 1
        
        elif direction == 0:
            print(frame_time)
            LOGGER.info(f"{'Exit_:':<14}{doorname, track}")
            if self.database:
                self.database.update_exit(1)
                self.exits = self.database.get_exit()
            else:
                self.exits += 1



    def getTypesDict(self, dicto):
        for key, elem in dicto.items():
            print("{} --> {}".format(key, type(elem)))

    def _mark_lost(self, trk_id):
        track = self.tracks.pop(trk_id)
        if track.confirmed:
            self.lost[trk_id] = track
            if len(self.lost) > self.history_size:
                self.lost.popitem(last=False)
    
    def _update_tracks(self):
        if not self.database:
            return
        db_tracks = self.database.get_tracks()
        # #print("Number 0f tracks=",len(db_tracks))
        # print(" DATBABSEA tracks=",db_tracks)
        if db_tracks:
            for track in db_tracks:
                if track["is_duplicate"] != 0: # 'is_duplicate ' --> the original id, 'trk_id' --> duplicate
                    dupl_track, locat_track = self.findTrk(track["is_duplicate"])
                    del_flag = False
                    if dupl_track == None or dupl_track.avg_feat.count == 0:
                        continue
                    if self.isExistTrkID(track["trk_id"], 0):
                        self.tracks[track["trk_id"]].merge_continuation(dupl_track)
                        self.tracks[track["is_duplicate"]] = self.tracks[track["trk_id"]]
                        del self.tracks[track["trk_id"]]
                        if locat_track != self.tracks:
                            del_flag = True
                    if self.isExistTrkID(track["trk_id"], 2):
                        self.lost[track["trk_id"]].merge_continuation(dupl_track)
                        self.lost[track["is_duplicate"]] = self.lost[track["trk_id"]]
                        del self.lost[track["trk_id"]]
                        if locat_track != self.lost:
                            del_flag = True
                    if self.isExistTrkID(track["trk_id"], 1):
                        # print("WE have {} of type {}".format( dupl_track.avg_feat.count, type( dupl_track.avg_feat.count)))
                        self.db_tracks[track["trk_id"]].merge_continuation(dupl_track)
                        self.db_tracks[track["is_duplicate"]] = self.db_tracks[track["trk_id"]]
                        del self.db_tracks[track["trk_id"]]
                        if locat_track != self.db_tracks:
                            del_flag = True
                    if del_flag:
                        del locat_track[track["is_duplicate"]]
                    self.database.update_idtrack(track["is_duplicate"], track["trk_id"], 0, self.camera)
                    # #print("Track {} is a duplicate...".format(track["trk_id"]))
                    continue

                if track["trk_id"] in self.duplicates:
                    # print("Track {} in duplicates".format(track["trk_id"]))
                    continue

                if self.isExistTrkID(track["trk_id"],0) or self.isExistTrkID(track["trk_id"],2):
                    continue
                if self.isExistTrkID(track["trk_id"],1):
                    #print("ID = {} Age = {} Hits = {}".format(track["trk_id"], track["age"], track["hits"]))
                    if int(track["hits"]) % 10 == 0:
                        state = track["mean"], track["covariance"]
                        dummy_track = Track(track["frame_id"], track["trk_id"], 
                                            np.asarray(track["tlbr"]), state,
                                            track["label"],avg= np.asarray(track["avg"]))
                                                
                        self.db_tracks[track["trk_id"]].merge_continuation(dummy_track)
                        # print("Track {} updated...".format(track["trk_id"]))
                        # pass
                    else:
                        continue
                track["frame_ids"] = deque(np.asarray(track["frame_ids"]), maxlen=self.buffer_size)
                track["bboxes"] = deque(np.asarray(track["bboxes"]), maxlen=self.buffer_size)
                track["tlbr"] = np.asarray(track["tlbr"])
                track["mean"] = np.asarray(track["mean"])
                track["covariance"] = np.asarray(track["covariance"])
                track["sum"] = np.asarray(track["sum"])
                track["avg"] = np.asarray(track["avg"])
                track["last_feat"] = np.asarray(track["last_feat"])
                track["keypoints"] = np.asarray(track["keypoints"],np.float32)
                track["prev_keypoints"] = np.asarray(track["prev_keypoints"],np.float32)
                track["label"] = np.int64(track["label"])
                state = track["mean"], track["covariance"]
                # track["direction"] = np.int64(track["direction"])
                # #print("SUM={} class={}".format(track["sum"],type(track["sum"])) )
                # #print("AVG={} class={}".format(track["avg"],type(track["avg"])) )
                # #print("Count={} class={}".format(track["count"],type(track["count"])) )

                db_track = Track(frame_id=track["frame_id"], trk_id=track["trk_id"],
                frame_ids=track["frame_ids"], bboxes=track["bboxes"], state=state, label=track["label"],
                tlbr=track["tlbr"], confirm_hits=track["confirm_hits"],age=track["age"], hits=track["hits"],
                _sum = track["sum"], avg=track["avg"], count=track["count"], last_feat=track["last_feat"],
                inlier_ratio=track["inlier_ratio"], keypoints=track["keypoints"], prev_keypoints=track["prev_keypoints"],
                direction=track["direction"])

                self.db_tracks[db_track.trk_id] = db_track
                # print(" DATABASE tracks= frame_id-->{}, trk_id-->{}".format(track["frame_id"], track["trk_id"]))
                # print("ADDED {} to DB tracks --> {}". format(db_track.trk_id, db_track))

    def _merge_tracks(self):
        return
        # print("+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+")

        lost_ids = [trk_id for trk_id, track in self.lost.items()
                        if track.avg_feat.count >= 2]
        db_ids = [trk_id for trk_id, track in self.db_tracks.items()
                        if track.avg_feat.count >= 2]                        
        lost_embeddings = [track.avg_feat() for trk_id, track in self.lost.items()
                    if track.avg_feat.count >= 2]
        db_embeddings = [track.avg_feat() for trk_id, track in self.db_tracks.items()
                    if track.avg_feat.count >= 2] 
        # u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]
        n_hist, n_det = len(lost_embeddings), len(db_embeddings)
        if n_hist == 0 or n_det == 0:
            return
        features = np.concatenate(lost_embeddings).reshape(n_hist, -1)
        embeddings = np.concatenate(db_embeddings).reshape(n_det, -1)

        cost = cdist(features, embeddings, self.metric)
        
        merge_matches, _, _ = greedy_match(cost, lost_ids, db_ids,
                                                        self.max_reid_cost, 1, "MERGER")
        
        # print("FEATURES SHAPE",features.shape)
        # print("EMBEDDINGS SHAPE",embeddings.shape)
        # print("MERGE_MATCHES",merge_matches)

        for lost_trk_id, db_trk_id in merge_matches:
            if db_trk_id < lost_trk_id:
                old_id = lost_trk_id
                new_id = db_trk_id
                old_track = self.lost.pop(lost_trk_id)
                new_track = self.db_tracks[db_trk_id]
            else:
                old_id = db_trk_id
                new_id = lost_trk_id
                old_track = self.db_tracks.pop(db_trk_id)
                new_track = self.lost[lost_trk_id]

            LOGGER.info(f"{'Imposter(MERGER):':<14}{old_track}")
            LOGGER.info(f"{'Original(MERGER):':<14}{new_track}")
            new_track.merge_continuation(old_track)
            new_track.direction = -1
            LOGGER.debug(f"{'Merged(MERGER):':<14}{old_id} -> {new_id}")
            if self.database:
                #attr_data = self.par.predict(crop(frame, new_track.tlbr))
                attr_data = None, None, None, None, None 
                self.database.insert_record(new_id, new_track.tlbr[2], new_track.tlbr[3], attr_data)
            self.duplicates.append(old_id)
            if new_track.hits >= 10 and self.database:
                pass
                self.database.update_record(new_id , old_id)
                self.database.update_idtrack(new_id, old_id, 1, self.camera)

    def _secmerge_tracks(self,boole=1):
        # print("+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+")
        # print("AC ->",self.tracks.items())
        # print("Lost ->",self.lost.items())
        # print("DB ->",self.db_tracks.items())

        if boole == 1:
            ac_ids = [trk_id for trk_id, track in self.tracks.items()
                            if track.avg_feat.count >= 2]                        
            lost_ids = [trk_id for trk_id, track in self.lost.items()
                            if track.avg_feat.count >= 2]
            ac_embeddings = [track.avg_feat() for trk_id, track in self.tracks.items()
                        if track.avg_feat.count >= 2] 
            lost_embeddings = [track.avg_feat() for trk_id, track in self.lost.items()
                        if track.avg_feat.count >= 2]
            # u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]
            # n_hist, n_det = len(lost_embeddings), len(ac_embeddings) 
            n_det, n_hist = len(ac_embeddings), len(lost_embeddings) 
            if n_hist == 0 or n_det == 0:
                # print("Null ac_embeddings or lost_embeddings!!!!!!")
                # print("Len ac_embeddings = ",n_det)
                # print("Len lost_embeddings = ",n_hist)
                return
            # features = np.concatenate(lost_embeddings).reshape(n_hist, -1)
            # embeddings = np.concatenate(ac_embeddings).reshape(n_det, -1)
            features = np.concatenate(ac_embeddings).reshape(n_det, -1)
            embeddings = np.concatenate(lost_embeddings).reshape(n_hist, -1)
            
            cost = cdist(features, embeddings, self.metric)
            
            merge_matches, _, _ = greedy_match(cost, ac_ids, lost_ids,
                                                            self.max_reid_cost, 1, "SECMERGER")

            # for lost_trk_id, ac_trk_id in merge_matches:
            for ac_trk_id, lost_trk_id in merge_matches:
                if ac_trk_id < lost_trk_id:
                    old_id = lost_trk_id
                    new_id = ac_trk_id
                else:
                    old_id = ac_trk_id
                    new_id = lost_trk_id
                old_track = self.lost.pop(lost_trk_id)
                new_track = self.tracks[ac_trk_id]

                # LOGGER.info(f"{'Imposter(SECMERGER):':<14}{old_track}")
                # LOGGER.info(f"{'Original(SECMERGER):':<14}{new_track}")
                old_id = max(old_track.trk_id,new_track.trk_id)
                new_id = min(old_track.trk_id,new_track.trk_id)
                new_track.merge_continuation(old_track)
                new_track.direction = 2
                self.tracks[new_id] = new_track
                self.tracks.pop(ac_trk_id)
                LOGGER.debug(f"{'Merged(SECMERGER):':<14}{old_id} -> {new_id}")
                if self.database:
                    self.database.update_record(new_id , old_id)
                    self.database.update_idtrack(new_id, old_id, 1, self.camera)
        elif boole == 2:
            # Re-id with DB tracks
            ac_ids = [trk_id for trk_id, track in self.tracks.items()
                            if track.avg_feat.count >= 2]                        
            db_ids = [trk_id for trk_id, track in self.db_tracks.items()
                            if track.avg_feat.count >= 2]                        
            ac_embeddings = [track.avg_feat() for trk_id, track in self.tracks.items()
                        if track.avg_feat.count >= 2] 
            db_embeddings = [track.avg_feat() for trk_id, track in self.db_tracks.items()
                        if track.avg_feat.count >= 2]
             # u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]
            n_det, n_hist = len(ac_embeddings), len(db_embeddings) 
            if n_hist == 0 or n_det == 0:
                # print("Null ac_embeddings or db_embeddings!!!!!!")
                # print("Len ac_embeddings = ",n_det)
                # print("Len db_embeddings = ",n_hist)
                return
            features = np.concatenate(ac_embeddings).reshape(n_det, -1)
            embeddings = np.concatenate(db_embeddings).reshape(n_hist, -1)
            
            cost = cdist(features, embeddings, self.metric)
            
            merge_matches, _, _ = greedy_match(cost, ac_ids, db_ids,
                                                            self.max_reid_cost, 1, "SECMERGER_DB")

            for ac_trk_id, db_trk_id in merge_matches:
                if ac_trk_id < db_trk_id:
                    old_id = db_trk_id
                    new_id = ac_trk_id
                else:
                    old_id = ac_trk_id
                    new_id = db_trk_id
                old_track = self.db_tracks.pop(db_trk_id)
                new_track = self.tracks[ac_trk_id]

                # LOGGER.info(f"{'Imposter(SECMERGER):':<14}{old_track}")
                # LOGGER.info(f"{'Original(SECMERGER):':<14}{new_track}")
                old_id = max(old_track.trk_id,new_track.trk_id)
                new_id = min(old_track.trk_id,new_track.trk_id)
                new_track.merge_continuation(old_track)
                new_track.direction = 2
                self.tracks[new_id] = new_track
                self.tracks.pop(ac_trk_id)
                LOGGER.debug(f"{'Merged(DB_SECMERGER):':<14}{old_id} -> {new_id}")
                if self.database:
                    self.database.update_record(new_id , old_id)
                    self.database.update_idtrack(new_id, old_id, 1, self.camera)


    def findTrk(self, trk_id):
        track = None
        location = None
        try:
            track = self.tracks[trk_id]
            location = self.tracks
        except (KeyError, IndexError):
            try:
                track = self.db_tracks[trk_id]
                location = self.db_tracks
            except (KeyError, IndexError):
                try:
                    track = self.lost[trk_id]
                    location = self.lost
                except (KeyError, IndexError):
                    return track, location
        finally:
            return track, location

    def isExistTrkID(self, trk_id, tracks=2):
        if tracks == 0:
            try:
                return self.tracks[trk_id] != None
            except KeyError:
                return False
        elif tracks == 1:
            try:
                return self.db_tracks[trk_id] != None 
            except KeyError:
                return False
        else:
            try:
                return self.lost[trk_id] != None
            except KeyError:
                return False

    def _group_tracks_by_depth(self, group_size=2):
        n_depth = (self.max_age + group_size) // group_size
        confirmed_by_depth = [[] for _ in range(n_depth)]
        unconfirmed = []
        # #print("n_depth={}, confirmed_by_depth={}".format(n_depth,confirmed_by_depth))
        for trk_id, track in self.tracks.items():
            if track.confirmed:
                depth = track.age // group_size
                # #print("age={}, depth={}".format(track.age,depth))
                confirmed_by_depth[depth].append(trk_id)
            else:
                unconfirmed.append(trk_id)
        return confirmed_by_depth, unconfirmed

    def _matching_cost(self, trk_ids, detections, embeddings, occluded_dmask):
        n_trk, n_det = len(trk_ids), len(detections)
        if n_trk == 0 or n_det == 0:
            return np.empty((n_trk, n_det))

        features = np.empty((n_trk, embeddings.shape[1]))
        invalid_fmask = np.zeros(n_trk, np.bool_)
        for i, trk_id in enumerate(trk_ids):
            track = self.tracks[trk_id]
            if track.avg_feat.is_valid():
                features[i, :] = track.avg_feat()
            else:
                invalid_fmask[i] = True

        empty_mask = invalid_fmask[:, None] | occluded_dmask
        fill_val = min(self.max_assoc_cost + 0.1, 1.)
        cost = cdist(features, embeddings, self.metric, empty_mask, fill_val)

        # fuse motion information
        for row, trk_id in enumerate(trk_ids):
            track = self.tracks[trk_id]
            m_dist = self.kf.motion_distance(*track.state, detections.tlbr)
            fuse_motion(cost[row], m_dist, self.motion_weight)

        # make sure associated pair has the same class label
        t_labels = np.fromiter((self.tracks[trk_id].label for trk_id in trk_ids), int, n_trk)
        gate_cost(cost, t_labels, detections.label, self.max_assoc_cost)
        return cost

    def _iou_cost(self, trk_ids, detections):
        n_trk, n_det = len(trk_ids), len(detections)
        if n_trk == 0 or n_det == 0:
            return np.empty((n_trk, n_det))

        t_labels = np.fromiter((self.tracks[trk_id].label for trk_id in trk_ids), int, n_trk)
        t_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids])
        d_bboxes = detections.tlbr
        iou_cost = iou_dist(t_bboxes, d_bboxes)
        gate_cost(iou_cost, t_labels, detections.label, 1. - self.iou_thresh)
        return iou_cost

    def _reid_cost(self, dico_ids, detections, embeddings, dic_num=1, dicto=None):
        if dic_num == 0:
            dico = self.tracks
        elif dic_num == 1:
            dico = self.lost
        elif dic_num == 2:
            dico = self.db_tracks
        else:
            dico = dicto
        n_hist, n_det = len(dico_ids), len(detections)
        if n_hist == 0 or n_det == 0:
            return np.empty((n_hist, n_det))
        
        lc = [dico[trk_id].avg_feat()
                                   for trk_id in dico_ids if dico[trk_id].avg_feat.count >= 2 ] #Cond for occluded tracks

        nones = [trk_id for trk_id in dico_ids if dico[trk_id].avg_feat() is None]
        if len(lc) > 0:
            n_hist = len(lc)
            if len(nones) > 0:
                for elem in nones:
                    del dico[elem]
            # print("lc concat",np.concatenate(lc).shape)
            features = np.concatenate(lc).reshape(n_hist, -1)
            # print("Feat_shape",features.shape)
            # print("embeddings_shape", embeddings.shape)
            cost = cdist(features, embeddings, self.metric)
            # print("COST ",cost)

            t_labels = np.fromiter((t.label for t in dico.values()), int, n_hist)
            gate_cost(cost, t_labels, detections.label)

            # print("LC[0] SHAPE",lc[0].shape)
            # print("FEATURES SHAPE",features.shape)
            # print("EMBEDDINGS SHAPE",embeddings.shape)
            # print("DETECTIONS SHAPE",detections.shape)
            # print("T_LABELS",t_labels)
            # print("GATE COST ",cost)
            # print("DETECTIONS STRAIGHT UP !",detections)
            return cost

    def _rectify_matches(self, matches, u_trk_ids, detections):
        matches, u_trk_ids = set(matches), set(u_trk_ids)
        #print("matches = {} u_trk_ids = {}".format(matches, u_trk_ids))
        inactive_matches = [match for match in matches if not self.tracks[match[0]].active]

        u_active = [trk_id for trk_id in u_trk_ids
                    if self.tracks[trk_id].confirmed and self.tracks[trk_id].active]
        #print("inactive_matches {} / u_active {}".format(inactive_matches, u_active))

        n_inactive_matches = len(inactive_matches)
        if n_inactive_matches == 0 or len(u_active) == 0:
            return matches, u_trk_ids
        m_inactive, det_ids = zip(*inactive_matches)
        t_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in u_active])
        d_bboxes = detections[det_ids,].tlbr
        iou_cost = iou_dist(t_bboxes, d_bboxes)

        col_indices = list(range(n_inactive_matches))
        dup_matches, _, _ = greedy_match(iou_cost, u_active, col_indices,
                                         1. - self.duplicate_thresh, 1, "Dupli-Match")
        #print("dup_matches = {}".format(dup_matches))
        for u_trk_id, col in dup_matches:
            m_trk_id, det_id = m_inactive[col], det_ids[col]
            t_u_active, t_m_inactive = self.tracks[u_trk_id], self.tracks[m_trk_id]
            if t_m_inactive.end_frame < t_u_active.start_frame:
                LOGGER.debug(f"{'Merged:':<14}{u_trk_id} -> {m_trk_id}")
                t_m_inactive.merge_continuation(t_u_active,False)
                u_trk_ids.remove(u_trk_id)
                del self.tracks[u_trk_id]
            else:
                LOGGER.debug(f"{'Duplicate:':<14}{m_trk_id} -> {u_trk_id}")
                u_trk_ids.remove(u_trk_id)
                u_trk_ids.add(m_trk_id)
                matches.remove((m_trk_id, det_id))
                matches.add((u_trk_id, det_id))
        return list(matches), list(u_trk_ids)

    def _remove_duplicate(self, trk_ids1, trk_ids2):
        if len(trk_ids1) == 0 or len(trk_ids2) == 0:
            return

        bboxes1 = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids1])
        bboxes2 = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids2])

        ious = bbox_ious(bboxes1, bboxes2)
        idx = np.where(ious >= self.duplicate_thresh)
        dup_ids = set()
        for row, col in zip(*idx):
            trk_id1, trk_id2 = trk_ids1[row], trk_ids2[col]
            track1, track2 = self.tracks[trk_id1], self.tracks[trk_id2]
            if len(track1) > len(track2):
                dup_ids.add(trk_id2)
            else:
                dup_ids.add(trk_id1)
        for trk_id in dup_ids:
            LOGGER.debug(f"{'Duplicate R:':<14}{self.tracks[trk_id]}")
            del self.tracks[trk_id]

    def getTimes(self):
        # timing results
        LOGGER.debug('=================Timing Stats=================')
        LOGGER.info(f"{'Average Preprocessing time:':<30}{Profiler.get_avg_millis('preproc'):>6.3f} ms")
        LOGGER.info(f"{'Average M1 time:':<30}{Profiler.get_avg_millis('m1'):>6.3f} ms")
        LOGGER.info(f"{'Average M2 time:':<30}{Profiler.get_avg_millis('m2'):>6.3f} ms")
        LOGGER.info(f"{'Average MREID time:':<30}{Profiler.get_avg_millis('mreid'):>6.3f} ms")
        LOGGER.info(f"{'Average MDB time:':<30}{Profiler.get_avg_millis('mdb'):>6.3f} ms")
        LOGGER.info(f"{'Average M3 time:':<30}{Profiler.get_avg_millis('m3'):>6.3f} ms")
        LOGGER.info(f"{'Average MREIDT time:':<30}{Profiler.get_avg_millis('mreidt'):>6.3f} ms")
        LOGGER.info(f"{'Average MRECTIFY time:':<30}{Profiler.get_avg_millis('mrect'):>6.3f} ms")
        LOGGER.info(f"{'Average MCOMBINE time:':<30}{Profiler.get_avg_millis('mcomb'):>6.3f} ms")
        LOGGER.info(f"{'Average R lost time:':<30}{Profiler.get_avg_millis('rlost'):>6.3f} ms")
        LOGGER.info(f"{'Average R db time:':<30}{Profiler.get_avg_millis('rdb'):>6.3f} ms")
        LOGGER.info(f"{'Average R reid time:':<30}{Profiler.get_avg_millis('rreid'):>6.3f} ms")
        LOGGER.info(f"{'Average R found time:':<30}{Profiler.get_avg_millis('rfound'):>6.3f} ms")
        LOGGER.info(f"{'Average Clean time:':<30}{Profiler.get_avg_millis('clean'):>6.3f} ms")
        LOGGER.info(f"{'Average New det time:':<30}{Profiler.get_avg_millis('new'):>6.3f} ms")
        # LOGGER.info(f"{'Average Aged time:':<30}{Profiler.get_avg_millis('aged'):>6.3f} ms")
