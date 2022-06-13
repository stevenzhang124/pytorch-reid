from collections import OrderedDict
import itertools
import logging
import numpy as np
import numba as nb
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps

from .track import Track
from .flow import Flow
from .kalman_filter import MeasType, KalmanFilter
from .utils.rect import as_rect, to_tlbr, iom, area


LOGGER = logging.getLogger(__name__)
CHI_SQ_INV_95 = 9.4877 # 0.95 quantile of chi-square distribution
INF_COST = 1e5


class MultiTracker:
    """
    Uses optical flow and Kalman filter to track multiple objects and
    associates detections to tracklets based on motion and appearance.
    Parameters
    ----------
    size : (int, int)
        Width and height of each frame.
    dt : float
        Time interval in seconds between each frame.
    metric : string
        Feature distance metric to associate tracklets. Usually
        `euclidean` or `cosine`.
    config : Dict
        Tracker hyperparameters.
    """

    def __init__(self, size, dt, metric, config, camera, database):
        self.size = size
        self.metric = metric
        self.max_age = config['max_age']
        self.age_factor = config['age_factor']
        self.motion_weight = config['motion_weight']
        self.max_feat_cost = config['max_feat_cost']
        self.max_reid_cost = config['max_reid_cost']
        self.iou_thresh = config['iou_thresh']
        self.duplicate_iou = config['duplicate_iou']
        self.conf_thresh = config['conf_thresh']
        self.lost_buf_size = config['lost_buf_size']

        self.database = database
        self.camera = camera
        self.cameraids = {"106":1, "113":2, "115":3, "116":4, "117":5, "118":6}

        self.next_id = self.database.get_nextid() #max(1,self.database.get_nextid())
        self.tracks = {}
        self.lost = OrderedDict()
        self.db_tracks = {}
        self.duplicates = []
        self.kf = KalmanFilter(dt, config['kalman_filter'])
        self.flow = Flow(self.size, config['flow'])
        self.frame_rect = to_tlbr((0, 0, *self.size))

        self.flow_bboxes = {}
        self.homography = None

    def initiate(self, frame, detections):
        """
        Initializes the tracker from detections in the first frame.
        Parameters
        ----------
        frame : ndarray
            Initial frame.
        detections : recarray[DET_DTYPE]
            Record array of N detections.
        """
        self.next_id = self.database.get_nextid()
        if self.tracks:
            self.tracks.clear()
        # Update tracks with database information and next_id to match database length
        self._update_tracks()
        self.flow.initiate(frame)
        for det in detections:
            state = self.kf.initiate(det.tlbr)
            new_trk = Track(frame_id=0, trk_id=self.next_id, tlbr=det.tlbr, state=state, label=det.label)
            self.tracks[self.next_id] = new_trk
            LOGGER.debug('Detected(init): %s', new_trk)
            self.database.update_nextid(self.next_id)
            self.next_id = max(1, self.database.get_nextid())

    def track(self, frame):
        """
        Convenience function  that combines `compute_flow` and `apply_kalman`.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """

        self.compute_flow(frame)
        self.apply_kalman()

    def compute_flow(self, frame):
        """
        Computes optical flow to estimate tracklet positions and camera motion.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        active_tracks = [track for track in self.tracks.values() if track.active]
        self.flow_bboxes, self.homography = self.flow.predict(frame, active_tracks)
        # if self.homography is None:
        #     # clear tracks when camera motion cannot be estimated
        #     self.tracks.clear()

    def apply_kalman(self,update=0):
        """
        Performs kalman filter prediction and update from flow measurements.
        The function should be called after `compute_flow`.
        """
        for trk_id, track in list(self.tracks.items()):
            mean, cov = track.state
            # mean, cov = self.kf.warp(mean, cov, self.homography)
            mean, cov = self.kf.predict(mean, cov)
            if trk_id in self.flow_bboxes:
                flow_tlbr = self.flow_bboxes[trk_id]
                # give large flow uncertainty for occluded tracks
                # usually these with high age and low inlier ratio
                std_multiplier = max(self.age_factor * track.age, 1) / track.inlier_ratio
                mean, cov = self.kf.update(mean, cov, flow_tlbr, MeasType.FLOW, std_multiplier)
            next_tlbr = as_rect(mean[:4])
            track.update(next_tlbr, (mean, cov))
            # if track.smooth_feature is not None and update==1:
            #     self.database.update_track(track.trk_id, track.trk_id, track._to_dict())
            if iom(next_tlbr, self.frame_rect) < 0.5:
                if track.confirmed:
                    # Add only "record" to DB
                    LOGGER.info('Lost(Out): %s', track)
                    self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3])
                    print("Track is {} ID is {}".format(track, trk_id))
                    # Delete "track" from DB
                    self._mark_lost(trk_id)
                else:
                    del self.tracks[trk_id]

    def update(self, frame_id, detections, embeddings):
        """
        Associates detections to tracklets based on motion and feature embeddings.
        Parameters
        ----------
        frame_id : int
            The next frame ID.
        detections : recarray[DET_DTYPE]
            Record array of N detections.
        embeddings : ndarray
            NxM matrix of N extracted embeddings with dimension M.
        """

        print("##############################################################################")

        self.next_id = max(self.next_id, self.database.get_nextid()) 

        det_ids = list(range(len(detections)))
        confirmed = [trk_id for trk_id, track in self.tracks.items() if track.confirmed]
        unconfirmed = [trk_id for trk_id, track in self.tracks.items() if not track.confirmed]
        print("CONFIRMED =", confirmed)
        print("UNCONFIRMED =", unconfirmed)
        print("LOST =", self.lost.keys())

        # # association with motion and embeddings
        # u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
        # cost = self._matching_cost(confirmed, u_detections, u_embeddings)
        # matches1, u_trk_ids1, u_det_ids = self._linear_assignment(cost, confirmed, u_det_ids)
        cost = self._matching_cost(confirmed, detections, embeddings)
        matches1, u_trk_ids1, u_det_ids = self._linear_assignment(cost, confirmed, det_ids)
        print("m1 -> u_det_ids",u_det_ids)

        # 2nd association with IoU
        active = [trk_id for trk_id in u_trk_ids1 if self.tracks[trk_id].active]
        u_trk_ids1 = [trk_id for trk_id in u_trk_ids1 if not self.tracks[trk_id].active]
        u_detections = detections[u_det_ids]
        cost = self._iou_cost(active, u_detections)
        matches2, u_trk_ids2, u_det_ids = self._linear_assignment(cost, active, u_det_ids, True)
        print("m2 -> u_det_ids",u_det_ids)

        # 3rd association with unconfirmed tracks
        u_detections = detections[u_det_ids]
        cost = self._iou_cost(unconfirmed, u_detections)
        matches3, u_trk_ids3, u_det_ids = self._linear_assignment(cost, unconfirmed, u_det_ids, True)
        print("m3 -> u_det_ids",u_det_ids)
        # if len(matches3) > 0:
        #     print("matches3 = {}, {}".format(type(matches3), matches3))

        # Re-id with lost tracks
        lost_ids = list(self.lost.keys())
        u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]
        u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
        cost = self._reid_cost(u_detections, u_embeddings)
        reid_matches, u_trk_ids4, u_det_ids = self._linear_assignment(cost, lost_ids, u_det_ids)
        print("reid_m -> u_det_ids",u_det_ids)


        # Re-id with DB tracks
        db_ids = list(self.db_tracks.keys())
        # u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]
        # u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
        prev_u = u_det_ids
        u_det_ids = [det_ids for trk_id, det_ids in matches3 if detections[det_ids].conf >= self.conf_thresh]
        print("db_m -> u_det_idsA",u_det_ids)
        u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
        cost = self._reid_cost(u_detections, u_embeddings, 1)
        db_matches, _, u_det_ids = self._linear_assignment(cost, db_ids, u_det_ids)
        print("db_m -> u_det_idsB",u_det_ids)
        if len(db_matches):
            print("db_matches = {}, {}".format(type(db_matches), db_matches))

        u_det_ids = prev_u


        # # Re-id with lost tracks
        # lost_ids = list(self.lost.keys())
        # u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]
        # u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
        # cost = self._reid_cost(u_detections, u_embeddings)
        # reid_matches, _, u_det_ids = self._linear_assignment(cost, lost_ids, u_det_ids)

        # matches = itertools.chain(matches0, matches1, matches2, matches3)
        # u_trk_ids = itertools.chain(u_trk_ids0, u_trk_ids1, u_trk_ids2, u_trk_ids3)
        matches = itertools.chain(matches1, matches2, matches3)
        u_trk_ids = itertools.chain(u_trk_ids1, u_trk_ids2, u_trk_ids3)
        updated, aged = [], []

        # print("MATCHES0 =", matches0)
        # if len(db_matches) > 0:
        #     print("DB_MATCHES =", db_matches)
        # if len(re_matches) > 0:
        #     print("RE_MATCHES =", re_matches)
        if len(matches1) > 0:
            print("MATCHES1 =", matches1)
        if len(matches2) > 0:
            print("MATCHES2 =", matches2)
        if len(matches3) > 0:
            print("MATCHES3 =", matches3)
        if len(reid_matches) > 0:
            print("REID_MATCHES =", reid_matches)
        if len(db_matches) > 0:
            print("DB_MATCHES =", db_matches)
        

        if len(u_trk_ids1) > 0:
            print("U_TRK_IDS_1 =", u_trk_ids1)
        if len(u_trk_ids2) > 0:
            print("U_TRK_IDS_2 =", u_trk_ids2)
        if len(u_trk_ids3) > 0:
            print("U_TRK_IDS_3 =", u_trk_ids3)
        if len(u_trk_ids4) > 0:
            print("U_TRK_IDS_3 =", u_trk_ids4)


        # update matched tracks
        for trk_id, det_id in matches:
            track = self.tracks[trk_id]
            det = detections[det_id]
            mean, cov = self.kf.update(*track.state, det.tlbr, MeasType.DETECTOR)
            next_tlbr = as_rect(mean[:4])
            track.update(next_tlbr, (mean, cov), embeddings[det_id])
            if track.hits == 1:
                LOGGER.info('Found: %s', track)
                self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3])
                # self.database.insert_track(trk_id, track._to_dict())
            if iom(next_tlbr, self.frame_rect) < 0.5:
                LOGGER.info('Out: %s', track)
                self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3])
                self._mark_lost(trk_id)
            else:
                if track.hits >= 10:
                    if track.hits == 10:
                        self.database.insert_track(trk_id, track._to_dict())
                    elif area(next_tlbr) > area(track.tlbr):
                        self.database.update_track(track.trk_id, track.trk_id, track._to_dict())
                updated.append(trk_id)
            

        # reactivate matched lost tracks
        for trk_id, det_id in reid_matches:
            track = self.lost[trk_id]
            det = detections[det_id]
            LOGGER.info('Re-identified: %s', track)
            state = self.kf.initiate(det.tlbr)
            track.reactivate(frame_id, det.tlbr, state, embeddings[det_id])
            self.tracks[trk_id] = track
            #Add record/track to DB
            self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3])
            # self.database.insert_track(trk_id, track._to_dict())
            del self.lost[trk_id]
            updated.append(trk_id)


        # reactivate matched db tracks
        for trk_id, det_id in db_matches:
            track = self.db_tracks[trk_id]
            det = detections[det_id]
            LOGGER.info('Re-tracked: %s', track)
            state = self.kf.initiate(det.tlbr)
            track.reactivate(frame_id, det.tlbr, state, embeddings[det_id])
            self.tracks[trk_id] = track
            self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3])
            # self.database.insert_track(trk_id, track._to_dict())
            del self.db_tracks[trk_id]
            updated.append(trk_id)


        # clean up lost tracks
        for trk_id in u_trk_ids:
            track = self.tracks[trk_id]
            if not track.confirmed:
                LOGGER.debug('Unconfirmed: %s', track)
                del self.tracks[trk_id]
                continue
            track.mark_missed()
            if track.age > self.max_age:
                LOGGER.info('Lost(Age): %s', track)
                # Remove from database
                # self.database.delete_record(trk_id)
                self._mark_lost(trk_id)
            else:
                LOGGER.info('Aged(likely lost in space): %s',track)
                aged.append(trk_id)

        # register new detections
        for det_id in u_det_ids:
            det = detections[det_id]
            state = self.kf.initiate(det.tlbr)
            # Add new_trk to db
            new_trk = Track(frame_id, self.next_id, det.tlbr, state, det.label)
            self.tracks[self.next_id] = new_trk
            LOGGER.debug('Detected: %s', new_trk)
            updated.append(self.next_id)
            self.database.update_nextid(self.next_id)
            self.next_id = max(self.next_id + 1, self.database.get_nextid())
            # self.next_id += 1

        # remove duplicate tracks
        print("UPDATED =", updated)
        print("AGED =", aged)

        self._remove_duplicate(updated, aged)

    def _mark_lost(self, trk_id):
        if self.tracks[trk_id].hits < 10:
            del self.tracks[trk_id]
            return
        self.lost[trk_id] = self.tracks[trk_id]
        # print("Lost: len={} keys={}".format(len(self.lost),self.lost.keys()))
        if len(self.lost) > self.lost_buf_size:
            self.lost.popitem(last=False)
            # self.database.delete_track(trk_id)
        del self.tracks[trk_id]

    def _remove_duplicate(self, updated, aged):
        if len(updated) == 0 or len(aged) == 0:
            return

        updated_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in updated])
        aged_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in aged])

        ious = bbox_overlaps(updated_bboxes, aged_bboxes)
        # print("iou_thresh = {}, ious = {}".format(self.duplicate_iou,ious))
        idx = np.where(ious >= self.duplicate_iou)
        dup_ids = set()
        for row, col in zip(*idx):
            updated_id, aged_id = updated[row], aged[col]
            if updated_id == aged_id:
                print("Updated/Aged_id are equal: {} and {}".format(updated_id, aged_id))
                continue
            # print("Updated_id={} Aged_id={}".format(updated_id,aged_id))
            if self.tracks[updated_id].start_frame <= self.tracks[aged_id].start_frame:
                dup_ids.add(aged_id)
            else:
                dup_ids.add(updated_id)
        for trk_id in dup_ids:
            LOGGER.debug('Duplicate: %s', self.tracks[trk_id])
            self.database.update_record(min(aged_id,updated_id), max(aged_id,updated_id))
            self.database.update_idtrack(min(aged_id,updated_id), max(aged_id,updated_id), 1, self.camera)
            self.duplicates.append(max(aged_id, updated_id))
            # self.database.update_track(min(aged_id,updated_id), max(aged_id,updated_id))
            # self.tracks[max(aged_id, updated_id)].start_frame = self.tracks[min(aged_id, updated_id)].start_frame
            # self.tracks[max(aged_id, updated_id)].trk_id = self.tracks[min(aged_id, updated_id)].trk_id
            # self.tracks[max(aged_id, updated_id)].hits = self.tracks[min(aged_id, updated_id)].hits
            # self.tracks[min(aged_id, updated_id)] = self.tracks[max(aged_id, updated_id)]
            del self.tracks[max(aged_id,updated_id)]
        

    def _update_tracks(self):
        db_tracks = self.database.get_tracks()
        # print("Number 0f tracks=",len(db_tracks))
        if db_tracks:
            for trck in db_tracks:
                if trck["is_duplicate"] != 0:
                    if self.isExistTrkID(trck["trk_id"], 0):
                        self.tracks[trck["trk_id"]].trk_id = trck["is_duplicate"]
                    if self.isExistTrkID(trck["trk_id"], 2):
                            del self.lost[trck["trk_id"]]
                    if self.isExistTrkID(trck["trk_id"], 1):
                            del self.db_tracks[trck["trk_id"]]

                    # # if original is in tracks or lost and duplicate is in db -> delete it
                    # if self.isExistTrkID(trck["is_duplicate"],0) or self.isExistTrkID(trck["trk_id"],2):
                    #     if self.isExistTrkID(trck["trk_id"],1):
                    #         del self.db_tracks[trck["trk_id"]]
                    #     else:
                    #         continue
                    # else:
                    #     if self.isExistTrkID(trck["trk_id"], 0):
                    #         self.tracks[trck["trk_id"]].trk_id = trck["is_duplicate"]
                    #     if self.isExistTrkID(trck["trk_id"], 2):
                    #         self.lost[trck["trk_id"]].trk_id = trck["is_duplicate"]
                    #     if self.isExistTrkID(trck["trk_id"], 1):
                    #         del self.db_tracks[trck["is_duplicate"]]
                    self.database.update_idtrack(trck["is_duplicate"], trck["trk_id"], 0, self.camera)
                    print("Track {} is a duplicate...".format(trck["trk_id"]))
                    continue

                if self.isExistTrkID(trck["trk_id"],0) or self.isExistTrkID(trck["trk_id"],1) or self.isExistTrkID(trck["trk_id"],2):
                    if self.isExistTrkID(trck["trk_id"],1):
                        print("ID = {} Age = {} Hits = {}".format(trck["trk_id"], trck["age"], trck["hits"]))
                    continue
                # if int(trck["is_duplicate"]) >= self.cameraids[str(self.camera)]:
                
                # print("Track_ID={}/{} will be checked...".format(trck["trk_id"],len(db_tracks)))
                trck["tlbr"] = np.array(trck["tlbr"])
                trck["mean"] = np.array(trck["mean"])
                trck["covariance"] = np.array(trck["covariance"])
                trck["smooth_feature"] = np.array(trck["smooth_feature"])
                trck["keypoints"] = np.array(trck["keypoints"],np.float32)
                trck["prev_keypoints"] = np.array(trck["prev_keypoints"],np.float32)
                trck["label"] = np.int64(trck["label"])
                state = trck["mean"], trck["covariance"]

                track = Track(frame_id=trck["frame_id"], trk_id=trck["trk_id"], tlbr=trck["tlbr"], state=state,
                label=trck["label"], age=trck["age"], hits=trck["hits"], alpha=trck["alpha"],
                smooth_feature=trck["smooth_feature"], inlier_ratio=trck["inlier_ratio"],
                keypoints=trck["keypoints"], prev_keypoints=trck["prev_keypoints"])

                self.db_tracks[track.trk_id] = track
                print("ADDED {} to DB tracks". format(track.trk_id))


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
        

    def _matching_cost(self, trk_ids, detections, embeddings, dict_type=0):
        dico = self.tracks if (dict_type == 0) else self.db_tracks 
        if len(trk_ids) == 0 or len(detections) == 0:
            return np.empty((len(trk_ids), len(detections)))

        features = [dico[trk_id].smooth_feature for trk_id in trk_ids]
        cost = cdist(features, embeddings, self.metric)
        for i, trk_id in enumerate(trk_ids):
            track = dico[trk_id]
            motion_dist = self.kf.motion_distance(*track.state, detections.tlbr)
            cost[i] = self._fuse_motion(cost[i], motion_dist, track.label, detections.label,
                                        self.max_feat_cost, self.motion_weight)
        return cost

    def _iou_cost(self, trk_ids, detections):
        if len(trk_ids) == 0 or len(detections) == 0:
            return np.empty((len(trk_ids), len(detections)))

        # make sure associated pair has the same class label
        trk_labels = np.array([self.tracks[trk_id].label for trk_id in trk_ids])
        trk_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids])
        det_bboxes = detections.tlbr
        ious = bbox_overlaps(trk_bboxes, det_bboxes)
        ious = self._gate_cost(ious, trk_labels, detections.label, self.iou_thresh, True)
        return ious

    def _reid_cost(self, detections, embeddings, dict_type=2):
        if dict_type == 0:
            dico = self.tracks
        elif dict_type == 1:
            dico = self.db_tracks
        else:
            dico = self.lost
        # dico = self.lost if (dict_type == 0) else self.db_tracks 
        if len(dico) == 0 or len(detections) == 0:
            return np.empty((len(dico), len(detections)))
        
        trk_labels = np.array([track.label for track in dico.values()])
        features = [track.smooth_feature for track in dico.values()]
        cost = cdist(features, embeddings, self.metric)
        cost = self._gate_cost(cost, trk_labels, detections.label, self.max_reid_cost)
        return cost
    
                    
    @staticmethod
    def _linear_assignment(cost, trk_ids, det_ids, maximize=False):
        rows, cols = linear_sum_assignment(cost, maximize)
        # print("rows={},cols={}".format(rows,cols))
        # print("trk_ids={},det_ids={}".format(trk_ids,det_ids))
        unmatched_rows = list(set(range(cost.shape[0])) - set(rows))
        unmatched_cols = list(set(range(cost.shape[1])) - set(cols))
        unmatched_trk_ids = [trk_ids[row] for row in unmatched_rows]
        unmatched_det_ids = [det_ids[col] for col in unmatched_cols]
        matches = []
        if not maximize:
            for row, col in zip(rows, cols):
                if cost[row, col] < INF_COST:
                    matches.append((trk_ids[row], det_ids[col]))
                else:
                    unmatched_trk_ids.append(trk_ids[row])
                    unmatched_det_ids.append(det_ids[col])
        else:
            for row, col in zip(rows, cols):
                if cost[row, col] > 0:
                    matches.append((trk_ids[row], det_ids[col]))
                else:
                    unmatched_trk_ids.append(trk_ids[row])
                    unmatched_det_ids.append(det_ids[col])
        return matches, unmatched_trk_ids, unmatched_det_ids

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _fuse_motion(cost, motion_dist, label, det_labels, max_cost, weight):
        gate = (cost > max_cost) | (motion_dist > CHI_SQ_INV_95) | (label != det_labels)
        cost = (1 - weight) * cost + weight * motion_dist
        cost[gate] = INF_COST
        return cost

    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _gate_cost(cost, trk_labels, det_labels, thresh, maximize=False):
        for i in nb.prange(len(cost)):
            if maximize:
                gate = (cost[i] < thresh) | (trk_labels[i] != det_labels)
                cost[i][gate] = 0
            else:
                gate = (cost[i] > thresh) | (trk_labels[i] != det_labels)
                cost[i][gate] = INF_COST
        return cost
