# from types import SimpleNamespace
# from collections import OrderedDict, deque
# import itertools
# import logging
# import numpy as np
# import numba as nb

# from ..track import Track
# from ..flow import Flow
# from ..kalman_filter import MeasType, KalmanFilter
# from .attribute_recognizer import AttributeRecognizer
# from ..distance import Metric, cdist, iou_dist
# from ..matching import linear_assignment, greedy_match, fuse_motion, gate_cost
# from ..rect import as_tlbr, to_tlbr, ios, bbox_ious, find_occluded, area, crop
# from .. import Profiler



# LOGGER = logging.getLogger(__name__)


# class MultiTracker:
#     def __init__(self, size, metric, camera, database, par,
#                  buffer_size=30,
#                  max_age=6,
#                  age_penalty=2,
#                  motion_weight=0.2,
#                  max_assoc_cost=0.9,
#                  max_reid_cost=0.45,
#                  iou_thresh=0.4,
#                  duplicate_thresh=0.8,
#                  occlusion_thresh=0.7,
#                  conf_thresh=0.5,
#                  confirm_hits=1,
#                  history_size=50,
#                  kalman_filter_cfg=None,
#                  flow_cfg=None):
#         """Class that uses KLT and Kalman filter to track multiple objects and
#         associates detections to tracklets based on motion and appearance.

#         Parameters
#         ----------
#         size : tuple
#             Width and height of each frame.
#         metric : {'euclidean', 'cosine'}
#             Feature distance metric to associate tracks.
#         max_age : int, optional
#             Max number of undetected frames allowed before a track is terminated.
#             Note that skipped frames are not included.
#         age_penalty : int, optional
#             Scale factor to penalize KLT measurements for tracks with large age.
#         motion_weight : float, optional
#             Weight for motion term in matching cost function.
#         max_assoc_cost : float, optional
#             Max matching cost for valid primary association.
#         max_reid_cost : float, optional
#             Max ReID feature dissimilarity for valid reidentification.
#         iou_thresh : float, optional
#             IoU threshold for association with unconfirmed and unmatched active tracks.
#         duplicate_thresh : float, optional
#             Track overlap threshold for removing duplicate tracks.
#         occlusion_thresh : float, optional
#             Detection overlap threshold for nullifying the extracted embeddings for association/reID.
#         conf_thresh : float, optional
#             Detection confidence threshold for starting a new track.
#         confirm_hits : int, optional
#             Min number of detections to confirm a track.
#         history_size : int, optional
#             Max size of track history to keep for reID.
#         kalman_filter_cfg : SimpleNamespace, optional
#             Kalman Filter configuration.
#         flow_cfg : SimpleNamespace, optional
#             Flow configuration.
#         """
#         self.size = size
#         self.metric = Metric[metric.upper()]
#         assert max_age >= 1
#         self.max_age = max_age
#         assert age_penalty >= 1
#         self.age_penalty = age_penalty
#         assert 0 <= motion_weight <= 1
#         self.motion_weight = motion_weight
#         assert 0 <= max_assoc_cost <= 2
#         self.max_assoc_cost = max_assoc_cost
#         assert 0 <= max_reid_cost <= 2
#         self.max_reid_cost = max_reid_cost
#         assert 0 <= iou_thresh <= 1
#         self.iou_thresh = iou_thresh
#         assert 0 <= duplicate_thresh <= 1
#         self.duplicate_thresh = duplicate_thresh
#         assert 0 <= occlusion_thresh <= 1
#         self.occlusion_thresh = occlusion_thresh
#         assert 0 <= conf_thresh <= 1
#         self.conf_thresh = conf_thresh
#         assert confirm_hits >= 1
#         self.confirm_hits = confirm_hits
#         assert history_size >= 0
#         self.history_size = history_size

#         if kalman_filter_cfg is None:
#             kalman_filter_cfg = SimpleNamespace()
#         if flow_cfg is None:
#             flow_cfg = SimpleNamespace()

#         self.database = database
#         self.camera = camera
#         self.cameraids = {"106":1, "113":2, "115":3, "116":4, "117":5, "118":6}

#         self.next_id = max(1, self.database.get_nextid()) if database else 1 
#         self.tracks = {}
#         self.lost = OrderedDict()
#         self.hist_tracks = OrderedDict()
#         self.db_tracks = {}
#         self.duplicates = []
#         self.kf = KalmanFilter(**vars(kalman_filter_cfg))
#         self.flow = Flow(self.size, **vars(flow_cfg))
#         self.frame_rect = to_tlbr((0, 0, *self.size))

#         self.klt_bboxes = {}
#         self.homography = None
#         self.buffer_size = buffer_size

#         self.par = AttributeRecognizer() if par else None

#     def reset(self, dt):
#         """Reset the tracker for new input context.

#         Parameters
#         ----------
#         dt : float
#             Time interval in seconds between each frame.
#         """
#         self.kf.reset_dt(dt)
#         self.lost.clear()
#         Track._count = 0

#     def init(self, frame, detections):
#         """Initializes the tracker from detections in the first frame.

#         Parameters
#         ----------
#         frame : ndarray
#             Initial frame.
#         detections : recarray[DET_DTYPE]
#             Record array of N detections.
#         """
#         self.next_id = self.database.get_nextid() if self.database else self.next_id
#         self.tracks.clear()
#         self._update_tracks()
#         self.flow.init(frame)
#         for det in detections:
#             state = self.kf.create(det.tlbr)
#             new_trk = Track(frame_id=0, trk_id=self.next_id, tlbr=det.tlbr, state=state, 
#                 label=det.label, confirm_hits=self.confirm_hits)
#             self.tracks[new_trk.trk_id] = new_trk
#             LOGGER.debug(f"{'Detected(Init):':<14}{new_trk}")

#     def track(self, frame):
#         """Convenience function that combines `compute_flow` and `apply_kalman`.

#         Parameters
#         ----------
#         frame : ndarray
#             The next frame.
#         """
#         self.compute_flow(frame)
#         self.apply_kalman(frame)

#     def compute_flow(self, frame):
#         """Computes optical flow to estimate tracklet positions and camera motion.

#         Parameters
#         ----------
#         frame : ndarray
#             The next frame.
#         """
#         active_tracks = [track for track in self.tracks.values() if track.active]
#         self.klt_bboxes, self.homography = self.flow.predict(frame, active_tracks)
#         # if self.homography is None:
#         #     # clear tracks when camera motion cannot be estimated
#         #     self.tracks.clear()

#     def apply_kalman(self, frame):
#         """Performs kalman filter predict and update from KLT measurements.
#         The function should be called after `compute_flow`.
#         """
#         for trk_id, track in list(self.tracks.items()):
#             mean, cov = track.state
#             # mean, cov = self.kf.warp(mean, cov, self.homography)
#             mean, cov = self.kf.predict(mean, cov)
#             if trk_id in self.klt_bboxes:
#                 klt_tlbr = self.klt_bboxes[trk_id]
#                 # give large KLT uncertainty for occluded tracks
#                 # usually these with large age and low inlier ratio
#                 std_multiplier = max(self.age_penalty * track.age, 1) / track.inlier_ratio
#                 mean, cov = self.kf.update(mean, cov, klt_tlbr, MeasType.FLOW, std_multiplier)
#             next_tlbr = as_tlbr(mean[:4])
#             track.update(next_tlbr, (mean, cov))
#             if ios(next_tlbr, self.frame_rect) < 0.5:
#                 if track.confirmed:
#                     LOGGER.info(f"{'Lost(Out):':<14}{track}")
#                     if self.database:
#                         #gender, age, hair, luggage, attire = self.par.predict(crop(frame,track.tlbr))
#                         gender, age, hair, luggage, attire = None, None, None, None, None 
#                         self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3], gender, age, hair, luggage, attire)
#                         #print("Track is {} ID is {}".format(track, trk_id))
#                 self._mark_lost(trk_id)
#                 # else:
#                 #     del self.tracks[trk_id]

#     def update(self, frame, frame_id, detections, embeddings):
#         """Associates detections to tracklets based on motion and feature embeddings.

#         Parameters
#         ----------
#         frame_id : int
#             The next frame ID.
#         detections : recarray[DET_DTYPE]
#             Record array of N detections.
#         embeddings : ndarray
#             NxM matrix of N extracted embeddings with dimension M.
#         """
#         with Profiler('preproc'):   
#             #print("###################################--{}--########################################".format(frame_id))

#             self.next_id = max(self.next_id, self.database.get_nextid())  if self.database else self.next_id

#             occluded_det_mask = find_occluded(detections.tlbr, self.occlusion_thresh)
#             confirmed_by_depth, unconfirmed = self._group_tracks_by_depth()
#             u_det_ids = list(range(len(detections)))
#             # LOGGER.info(f"{'Detections:':<14}{u_det_ids}")
#             # LOGGER.info(f"{'Occluded:':<14}{occluded_det_mask}")
#             # LOGGER.info(f"{'Confirmed:':<14}{confirmed_by_depth}")
#             # LOGGER.info(f"{'Unconfirmed:':<14}{unconfirmed}")
#             # LOGGER.info(f"{'Lost:':<14}{self.lost.keys()}")

#             # #print("DETECTIONS=",u_det_ids)
#             # #print("OCCLUDED =",occluded_det_mask)
#             # #print("CONFIRMED =", confirmed_by_depth)
#             # #print("UNCONFIRMED =", unconfirmed)
#             # #print("LOST =", self.lost.keys())

#             # association with motion and embeddings, tracks with small age are prioritized
#             matches1 = []
#             u_trk_ids1 = []

#         with Profiler('m1'):
#             for depth, trk_ids in enumerate(confirmed_by_depth):
#                 # #print("depth = {} , trk_ids = {} ".format(depth,trk_ids))
#                 if len(u_det_ids) == 0:
#                     u_trk_ids1.extend(itertools.chain.from_iterable(confirmed_by_depth[depth:]))
#                     break
#                 if len(trk_ids) == 0:
#                     continue
#                 u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
#                 u_occluded_dmask = occluded_det_mask[u_det_ids]
#                 cost = self._matching_cost(trk_ids, u_detections, u_embeddings, u_occluded_dmask)
#                 matches, u_trk_ids, u_det_ids = linear_assignment(cost, trk_ids, u_det_ids, 1, "Matches1")
#                 matches1 += matches
#                 u_trk_ids1 += u_trk_ids

#         with Profiler('m2'):   
#             # 2nd association with IoU
#             active = [trk_id for trk_id in u_trk_ids1 if self.tracks[trk_id].active]
#             u_trk_ids1 = [trk_id for trk_id in u_trk_ids1 if not self.tracks[trk_id].active]
#             u_detections = detections[u_det_ids]
#             cost = self._iou_cost(active, u_detections)
#             matches2, u_trk_ids2, u_det_ids = linear_assignment(cost, active, u_det_ids, 1, "Matches2")

#         with Profiler('mreid'):   
#             # Re-id with lost tracks
#             lost_ids = [trk_id for trk_id, track in self.lost.items()
#                         if track.avg_feat.count >= 2]
#             u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]
#             u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
#             cost = self._reid_cost(lost_ids, u_detections, u_embeddings, 1)
#             lost_matches, u_trk_ids4, u_det_idsL = greedy_match(cost, lost_ids, u_det_ids,
#                                                             self.max_reid_cost, 1, "Lost")
        
#         with Profiler('mdb'):   
#             # Re-id with DB tracks
#             db_ids = list(self.db_tracks.keys())
#             # u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
#             cost = self._reid_cost(db_ids, u_detections, u_embeddings, 2)
#             db_matches, u_trk_ids5, u_det_idsD = greedy_match(cost, db_ids, u_det_ids,
#                                                             self.max_reid_cost, 1, "DB")

#             # if len(u_det_ids) > 0:
#                 #print("PRE U_DET_IDS=",u_det_ids)

#             u_det_ids = list(set(u_det_idsL).intersection(u_det_idsD))

#             # if len(u_det_ids) > 0:
#                 #print("POST U_DET_IDS=",u_det_ids)

#             _to_remove_lost = []
#             _to_remove_db = []

#         with Profiler('m3'):   
#             # 3rd association with unconfirmed tracks
#             u_detections = detections[u_det_ids]
#             cost = self._iou_cost(unconfirmed, u_detections)
#             matches3, u_trk_ids3, u_det_ids = linear_assignment(cost, unconfirmed, u_det_ids, 1, "Matches3")

#         with Profiler('mreidt'):   
#             # reID with track history
#             lost_ids = [trk_id for trk_id, track in self.lost.items()
#                         if track.avg_feat.count >= 2]
#             u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]
#             valid_u_det_ids = [det_id for det_id in u_det_ids if not occluded_det_mask[det_id]]
#             invalid_u_det_ids = [det_id for det_id in u_det_ids if occluded_det_mask[det_id]]
#             u_detections, u_embeddings = detections[valid_u_det_ids], embeddings[valid_u_det_ids]
#             cost = self._reid_cost(lost_ids, u_detections, u_embeddings, 1)
#             reid_matches, _, reid_u_det_ids = greedy_match(cost, lost_ids, valid_u_det_ids,
#                                                        self.max_reid_cost, 1, "Reid-Matches")

#             matches = itertools.chain(matches1, matches2, matches3)
#             u_trk_ids = itertools.chain(u_trk_ids1, u_trk_ids2, u_trk_ids3)
        
#         with Profiler('mrect'):   
#             # rectify matches that may cause duplicate tracks
#             matches, u_trk_ids = self._rectify_matches(matches, u_trk_ids, detections)
#             updated_found = []

#         with Profiler('mcomb'):   
#             if len(lost_matches) > 0:
#                 for db_trk_id, det_id in db_matches:
#                     lost_trk_id = [trk_id1 for trk_id1, det_id1 in lost_matches if det_id == det_id1]
#                     #print("COMBY TRK=",lost_trk_id)
#                     if len(lost_trk_id) == 0:
#                         continue
#                     else:
#                         pass
#                         #print("LOST_MATCHES =", lost_matches)
#                         #print("DB_MATCHES =", db_matches)
#                     lost_trk_id = lost_trk_id[0]
                    
#                     if db_trk_id < lost_trk_id:
#                         old_id = lost_trk_id
#                         new_id = db_trk_id
#                         old_track = self.lost.pop(lost_trk_id)
#                         new_track = self.db_tracks.pop(db_trk_id)
#                     else:
#                         old_id = db_trk_id
#                         new_id = lost_trk_id
#                         old_track = self.db_tracks.pop(db_trk_id)
#                         new_track = self.lost.pop(lost_trk_id)

#                     _to_remove_lost.append((lost_trk_id,det_id))
#                     _to_remove_db.append((db_trk_id,det_id))
                
#                     # #print("ELEMENTS: old_id={}, new_id={}, old_track={}, new_track={}".format(old_id, new_id, old_track, new_track))

#                     det = detections[det_id]
#                     LOGGER.info(f"{'Imposter:':<14}{old_track}")
#                     state = self.kf.create(det.tlbr)
#                     new_track.reinstate(frame_id, det.tlbr, state, embeddings[det_id])
#                     new_track.merge_continuation(old_track)
#                     self.tracks[new_id] = new_track
#                     LOGGER.debug(f"{'Merged:':<14}{old_id} -> {new_id}")
#                     if self.database:
#                         #gender, age, hair, luggage, attire = self.par.predict(crop(frame, new_track.tlbr))
#                         gender, age, hair, luggage, attire = None, None, None, None, None 
#                         self.database.insert_record(new_id, new_track.tlbr[2], new_track.tlbr[3], gender, age, hair, luggage, attire)
#                     self.duplicates.append(old_id)
#                     if self.tracks[new_id].hits >= 10 and self.database:
#                         pass
#                         self.database.update_record(new_id , old_id)
#                         self.database.update_idtrack(new_id, old_id, 1, self.camera)


#                 # #print("_to_remove_db=",_to_remove_db)
#                 # #print("_to_remove_lost=",_to_remove_lost)
#                 for k,v in _to_remove_db:
#                     db_matches.remove((k,v))
#                 for k,v in _to_remove_lost:
#                     lost_matches.remove((k,v))
        
#         with Profiler('rlost'):   
#             # reactivate matched lost tracks
#             for trk_id, det_id in lost_matches:
#                 if not self.isExistTrkID(trk_id,2):
#                     continue
#                 track = self.lost.pop(trk_id)
#                 det = detections[det_id]
#                 LOGGER.info(f"{'Re-identified:':<14}{track}")
#                 updated_found.append((trk_id, det_id))
#                 state = self.kf.create(det.tlbr)
#                 track.reinstate(frame_id, det.tlbr, state, embeddings[det_id])
#                 self.tracks[trk_id] = track
#                 if self.database:
#                     #gender, age, hair, luggage, attire = self.par.predict(crop(frame,track.tlbr))
#                     gender, age, hair, luggage, attire = None, None, None, None, None 
#                     self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3], gender, age, hair, luggage, attire)

#         with Profiler('rdb'):   
#             # reactivate matched db tracks
#             for trk_id, det_id in db_matches:
#                 if not self.isExistTrkID(trk_id,1):
#                     continue
#                 track = self.db_tracks.pop(trk_id)
#                 det = detections[det_id]
#                 LOGGER.info(f"{'Re-tracked:':<14}{track}")
#                 updated_found.append((trk_id, det_id))
#                 state = self.kf.create(det.tlbr)
#                 track.reinstate(frame_id, det.tlbr, state, embeddings[det_id])
#                 self.tracks[trk_id] = track
#                 if self.database:
#                     #gender, age, hair, luggage, attire = self.par.predict(crop(frame,track.tlbr))
#                     gender, age, hair, luggage, attire = None, None, None, None, None 
#                     self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3], gender, age, hair, luggage, attire)

#         with Profiler('rreid'):   
#             # reinstate matched tracks
#             for trk_id, det_id in reid_matches:
#                 if not self.isExistTrkID(trk_id):
#                     continue
#                 track = self.lost.pop(trk_id)
#                 det = detections[det_id]
#                 LOGGER.info(f"{'Reidentified:':<14}{track}")
#                 updated_found.append((trk_id, det_id))
#                 state = self.kf.create(det.tlbr)
#                 track.reinstate(frame_id, det.tlbr, state, embeddings[det_id])
#                 self.tracks[trk_id] = track
#                 if self.database:
#                     #gender, age, hair, luggage, attire = self.par.predict(crop(frame,track.tlbr))
#                     gender, age, hair, luggage, attire = None, None, None, None, None 
#                     self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3], gender, age, hair, luggage, attire)
        
#         with Profiler('rfound'):   
#             # update matched tracks
#             for trk_id, det_id in matches:
#                 track = self.tracks[trk_id]
#                 det = detections[det_id]
#                 mean, cov = self.kf.update(*track.state, det.tlbr, MeasType.DETECTOR)
#                 next_tlbr = as_tlbr(mean[:4])
#                 is_valid = not occluded_det_mask[det_id]
#                 track.add_detection(frame_id, next_tlbr, (mean, cov), embeddings[det_id], is_valid)
#                 if track.hits == self.confirm_hits or ios(next_tlbr, self.frame_rect) < 0.5 :
#                     if track.hits == self.confirm_hits:
#                         LOGGER.info(f"{'Found:':<14}{track}")
#                         updated_found.append((trk_id, det_id))
#                     if ios(next_tlbr, self.frame_rect) < 0.5:
#                         is_valid = False
#                         if track.confirmed:
#                             LOGGER.info(f"{'Out:':<14}{track}")
#                         self._mark_lost(trk_id)
#                     if self.database:
#                         gender, age, hair, luggage, attire = self.par.predict(crop(frame,track.tlbr)) if sel.par else None, None, None, None, None 
#                         self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3], gender, age, hair, luggage, attire)
#                 else:
#                     if track.hits >= 10 and self.database:
#                         if track.hits == 10:
#                             pass
#                             # gender, age, hair, luggage, attire = self.par.predict(crop(frame,track.tlbr))
#                             gender, age, hair, luggage, attire = None, None, None, None, None 
#                             self.database.insert_record(trk_id, track.tlbr[2], track.tlbr[3], gender, age, hair, luggage, attire)
#                             self.database.insert_track(trk_id, track._to_dict())
#                         elif track.hits % 10 == 0:
#                             pass
#                             self.database.update_track(track.trk_id, track.trk_id, track._to_dict())
        
#         with Profiler('clean'):   
#             # clean up lost tracks
#             for trk_id in u_trk_ids:
#                 track = self.tracks[trk_id]
#                 track.mark_missed()
#                 if not track.confirmed:
#                     LOGGER.debug(f"{'Unconfirmed:':<14}{track}")
#                     del self.tracks[trk_id]
#                     continue
#                 if track.age > self.max_age:
#                     LOGGER.info(f"{'Lost(Age):':<14}{track}")
#                     self._mark_lost(trk_id)

#             # if len(lost_matches) > 0:
#             #     #print("LOST_MATCHES =", lost_matches)
#             # if len(db_matches) > 0:
#             #     #print("DB_MATCHES =", db_matches)
#             # if len(matches1) > 0:
#             #     #print("MATCHES1 =", matches1)
#             # if len(matches2) > 0:
#             #     #print("MATCHES2 =", matches2)
#             # if len(matches3) > 0:
#             #     #print("MATCHES3 =", matches3)
#             # if len(reid_matches) > 0:
#             #     #print("REID_MATCHES =", reid_matches)
#             # # if len(list(matches)) > 0:
#             # #     #print("Post-MATCHES=", list(matches))

#             # if len(u_trk_ids1) > 0:
#             #     #print("U_TRK_IDS_1 =", u_trk_ids1)
#             # if len(u_trk_ids2) > 0:
#             #     #print("U_TRK_IDS_2 =", u_trk_ids2)
#             # if len(u_trk_ids3) > 0:
#             #     #print("U_TRK_IDS_3 =", u_trk_ids3)
#             # # if len(list(u_trk_ids)) > 0:
#             # #     #print("Post-U_TRK_IDS = ", list(u_trk_ids))

#         with Profiler('new'):   
#             # start new tracks
#             u_det_ids = itertools.chain(invalid_u_det_ids, reid_u_det_ids)
#             for det_id in u_det_ids:
#                 det = detections[det_id]
#                 state = self.kf.create(det.tlbr)
#                 new_trk = Track(frame_id=frame_id, trk_id=self.next_id, tlbr=det.tlbr,
#                  state=state, label=det.label, confirm_hits=self.confirm_hits)
#                 self.tracks[new_trk.trk_id] = new_trk
#                 LOGGER.debug(f"{'Detected:':<14}{new_trk}")
#                 if self.database:
#                     self.database.update_nextid(self.next_id)
#                 self.next_id = max(self.next_id + 1, self.database.get_nextid()) if self.database else self.next_id + 1
            
#             aged = [(trk_id,trk_id) for trk_id in u_trk_ids
#                         if self.isExistTrkID(trk_id, 0) and self.tracks[trk_id].confirmed and self.tracks[trk_id].active]
#             aged_trk = [trk_id for trk_id,det_id in aged]
#             #print("AGED ={}".format(aged))
#             #print("UPDATED ={}".format(updated_found))
        
#         with Profiler('aged'):   
#             # Reid aged
#             c_det_ids = [det_id for trk_id, det_id in updated_found]
#             u_detections, u_embeddings = detections[c_det_ids], embeddings[c_det_ids]
#             aged_all = {}
#             for trk_id1, det_id1 in aged:
#                 aged_all = {trk_id:track for trk_id, track in self.tracks.items() if trk_id == trk_id1}
#             cost_aged = self._reid_cost(aged_trk, u_detections, u_embeddings, 3, aged_all)
#             aged_matches, _, _ = greedy_match(cost_aged, aged_trk, c_det_ids, self.max_reid_cost, 1, "Reid-Aged")
#             # if len(aged_matches) > 0:
#                 #print("AGED_MATCHES =", aged_matches)

#             for aged_trk_id, det_id in aged_matches:
#                 lost_trk_id = [trk_id1 for trk_id1, det_id1 in lost_matches if det_id == det_id1]
#                 #print("MATCHES TRK=",lost_trk_id)
#                 if len(lost_trk_id) == 0:
#                     continue
#                 lost_trk_id = lost_trk_id[0]

#                 if aged_trk_id < lost_trk_id:
#                     old_id = lost_trk_id
#                     new_id = aged_trk_id
#                 else:
#                     old_id = aged_trk_id
#                     new_id = lost_trk_id
#                 old_track = self.tracks.pop(old_id)
#                 new_track = self.tracks[new_id]

#                 det = detections[det_id]
#                 LOGGER.info(f"{'Redundant:':<14}{old_track}")
#                 state = self.kf.create(det.tlbr)
#                 track.reinstate(frame_id, det.tlbr, state, embeddings[det_id])
#                 new_track.merge_continuation(old_track)
#                 self.tracks[new_id] = new_track
#                 LOGGER.debug(f"{'Merged:':<14}{old_id} -> {new_id}")
#                 if self.database:
#                     #gender, age, hair, luggage, attire = self.par.predict(crop(frame,new_track.tlbr))
#                     gender, age, hair, luggage, attire = None, None, None, None, None 
#                     self.database.insert_record(new_id, new_track.tlbr[2], new_track.tlbr[3], gender, age, hair, luggage, attire)
#                     self.duplicates.append(old_id)
#                 if new_track.hits >= 10 and self.database:
#                     pass
#                     self.database.update_record(new_id ,old_id)
#                     self.database.update_idtrack(new_id, old_id, 1, self.camera)

#         # with Profiler('preproc'):   
#         #     occluded_det_mask = find_occluded(detections.tlbr, self.occlusion_thresh)
#         #     confirmed_by_depth, unconfirmed = self._group_tracks_by_depth()

#         #     # association with motion and embeddings, tracks with small age are prioritized
#         #     matches1 = []
#         #     u_trk_ids1 = []
#         #     u_det_ids = list(range(len(detections)))
        
#         # with Profiler('m1'):
#         #     for depth, trk_ids in enumerate(confirmed_by_depth):
#         #         if len(u_det_ids) == 0:
#         #             u_trk_ids1.extend(itertools.chain.from_iterable(confirmed_by_depth[depth:]))
#         #             break
#         #         if len(trk_ids) == 0:
#         #             continue
#         #         u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
#         #         u_occluded_dmask = occluded_det_mask[u_det_ids]
#         #         cost = self._matching_cost(trk_ids, u_detections, u_embeddings, u_occluded_dmask)
#         #         matches, u_trk_ids, u_det_ids = linear_assignment(cost, trk_ids, u_det_ids)
#         #         matches1 += matches
#         #         u_trk_ids1 += u_trk_ids
        
#         # with Profiler('m2'):
#         #     # 2nd association with IoU
#         #     active = [trk_id for trk_id in u_trk_ids1 if self.tracks[trk_id].active]
#         #     u_trk_ids1 = [trk_id for trk_id in u_trk_ids1 if not self.tracks[trk_id].active]
#         #     u_detections = detections[u_det_ids]
#         #     cost = self._iou_cost(active, u_detections)
#         #     matches2, u_trk_ids2, u_det_ids = linear_assignment(cost, active, u_det_ids)
        
#         # with Profiler('m3'):
#         #     # 3rd association with unconfirmed tracks
#         #     u_detections = detections[u_det_ids]
#         #     cost = self._iou_cost(unconfirmed, u_detections)
#         #     matches3, u_trk_ids3, u_det_ids = linear_assignment(cost, unconfirmed, u_det_ids)
        
#         # with Profiler('mreidt'):   
#         #     # reID with track history
#         #     hist_ids = [trk_id for trk_id, track in self.hist_tracks.items()
#         #                 if track.avg_feat.count >= 2]

#         #     u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]
#         #     valid_u_det_ids = [det_id for det_id in u_det_ids if not occluded_det_mask[det_id]]
#         #     invalid_u_det_ids = [det_id for det_id in u_det_ids if occluded_det_mask[det_id]]

#         #     u_detections, u_embeddings = detections[valid_u_det_ids], embeddings[valid_u_det_ids]
#         #     cost = self._reid_cost(hist_ids, u_detections, u_embeddings)

#         #     reid_matches, _, reid_u_det_ids = greedy_match(cost, hist_ids, valid_u_det_ids,
#         #                                                    self.max_reid_cost)

#         #     matches = itertools.chain(matches1, matches2, matches3)
#         #     u_trk_ids = itertools.chain(u_trk_ids1, u_trk_ids2, u_trk_ids3)

#         # with Profiler('mrect'):   
#         #     # rectify matches that may cause duplicate tracks
#         #     matches, u_trk_ids = self._rectify_matches(matches, u_trk_ids, detections)
        
#         # with Profiler('rreid'):   
#         #     # reinstate matched tracks
#         #     for trk_id, det_id in reid_matches:
#         #         track = self.hist_tracks.pop(trk_id)
#         #         det = detections[det_id]
#         #         LOGGER.info(f"{'Reidentified:':<14}{track}")
#         #         state = self.kf.create(det.tlbr)
#         #         track.reinstate(frame_id, det.tlbr, state, embeddings[det_id])
#         #         self.tracks[trk_id] = track
        
#         # with Profiler('rfound'):   
#         #     # update matched tracks
#         #     for trk_id, det_id in matches:
#         #         track = self.tracks[trk_id]
#         #         det = detections[det_id]
#         #         mean, cov = self.kf.update(*track.state, det.tlbr, MeasType.DETECTOR)
#         #         next_tlbr = as_tlbr(mean[:4])
#         #         is_valid = not occluded_det_mask[det_id]
#         #         if track.hits == self.confirm_hits - 1:
#         #             LOGGER.info(f"{'Found:':<14}{track}")
#         #         if ios(next_tlbr, self.frame_rect) < 0.5:
#         #             is_valid = False
#         #             if track.confirmed:
#         #                 LOGGER.info(f"{'Out:':<14}{track}")
#         #             self._mark_lost(trk_id)
#         #         track.add_detection(frame_id, next_tlbr, (mean, cov), embeddings[det_id], is_valid)
        
#         # with Profiler('clean'):   
#         #     # clean up lost tracks
#         #     for trk_id in u_trk_ids:
#         #         track = self.tracks[trk_id]
#         #         track.mark_missed()
#         #         if not track.confirmed:
#         #             LOGGER.debug(f"{'Unconfirmed:':<14}{track}")
#         #             del self.tracks[trk_id]
#         #             continue
#         #         if track.age > self.max_age:
#         #             LOGGER.info(f"{'Lost:':<14}{track}")
#         #             self._mark_lost(trk_id)
        
#         # with Profiler('new'):   
#             u_det_ids = itertools.chain(invalid_u_det_ids, reid_u_det_ids)
#             # start new tracks
#             for det_id in u_det_ids:
#                 det = detections[det_id]
#                 state = self.kf.create(det.tlbr)
#                 new_trk = Track(frame_id, self.next_id, det.tlbr, state, det.label, self.confirm_hits)
#                 # print("new trck is = ", new_trk)
#                 self.tracks[new_trk.trk_id] = new_trk

#                 # new_trk = Track(frame_id=frame_id, trk_id=self.next_id, tlbr=det.tlbr,
#                 #  state=state, label=det.label, confirm_hits=self.confirm_hits)
#                 LOGGER.debug(f"{'Detected:':<14}{new_trk}")

#         # LOGGER.info(f"{'Preprocessing time:':<30}{Profiler.get_avg_millis('preproc'):>6.3f} ms")
#         # LOGGER.info(f"{'M1 time:':<30}{Profiler.get_avg_millis('m1'):>6.3f} ms")
#         # LOGGER.info(f"{'M2 time:':<30}{Profiler.get_avg_millis('m2'):>6.3f} ms")
#         # LOGGER.info(f"{'MREID time:':<30}{Profiler.get_avg_millis('mreid'):>6.3f} ms")
#         # LOGGER.info(f"{'MDB time:':<30}{Profiler.get_avg_millis('mdb'):>6.3f} ms")
#         # LOGGER.info(f"{'M3 time:':<30}{Profiler.get_avg_millis('m3'):>6.3f} ms")
#         # LOGGER.info(f"{'MREIDT time:':<30}{Profiler.get_avg_millis('mreidt'):>6.3f} ms")
#         # LOGGER.info(f"{'MRECTIFY time:':<30}{Profiler.get_avg_millis('mrect'):>6.3f} ms")
#         # LOGGER.info(f"{'MCOMBINE time:':<30}{Profiler.get_avg_millis('mcomb'):>6.3f} ms")
#         # LOGGER.info(f"{'R lost time:':<30}{Profiler.get_avg_millis('rlost'):>6.3f} ms")
#         # LOGGER.info(f"{'R db time:':<30}{Profiler.get_avg_millis('rdb'):>6.3f} ms")
#         # LOGGER.info(f"{'R reid time:':<30}{Profiler.get_avg_millis('rreid'):>6.3f} ms")
#         # LOGGER.info(f"{'R found time:':<30}{Profiler.get_avg_millis('rfound'):>6.3f} ms")
#         # LOGGER.info(f"{'Clean time:':<30}{Profiler.get_avg_millis('clean'):>6.3f} ms")
#         # LOGGER.info(f"{'New det time:':<30}{Profiler.get_avg_millis('new'):>6.3f} ms")
#         # LOGGER.info(f"{'Aged time:':<30}{Profiler.get_avg_millis('aged'):>6.3f} ms")

#     def getTypesDict(self, dicto):
#         for key, elem in dicto.items():
#             print("{} --> {}".format(key, type(elem)))

#     def _mark_lost(self, trk_id):
#         track = self.tracks.pop(trk_id)
#         if track.confirmed:
#             self.lost[trk_id] = track
#             if len(self.lost) > self.history_size:
#                 self.lost.popitem(last=False)
    
#     # @nb.njit(fastmath=True, cache=True)
#     def _update_tracks(self):
#         if not self.database:
#             return
#         db_tracks = self.database.get_tracks()
#         # #print("Number 0f tracks=",len(db_tracks))
#         if db_tracks:
#             for track in db_tracks:
#                 if track["is_duplicate"] != 0: # 'is_duplicate ' --> the original id, 'trk_id' --> duplicate
#                     dupl_track, locat_track = self.findTrk(track["is_duplicate"])
#                     del_flag = False
#                     if dupl_track == None:
#                         continue
#                     if self.isExistTrkID(track["trk_id"], 0):
#                         self.tracks[track["trk_id"]].merge_continuation(dupl_track)
#                         self.tracks[track["is_duplicate"]] = self.tracks[track["trk_id"]]
#                         del self.tracks[track["trk_id"]]
#                         if locat_track != self.tracks:
#                             del_flag = True
#                     if self.isExistTrkID(track["trk_id"], 2):
#                         self.lost[track["trk_id"]].merge_continuation(dupl_track)
#                         self.lost[track["is_duplicate"]] = self.lost[track["trk_id"]]
#                         del self.lost[track["trk_id"]]
#                         if locat_track != self.lost:
#                             del_flag = True
#                     if self.isExistTrkID(track["trk_id"], 1):
#                         self.db_tracks[track["trk_id"]].merge_continuation(dupl_track)
#                         self.db_tracks[track["is_duplicate"]] = self.db_tracks[track["trk_id"]]
#                         del self.db_tracks[track["trk_id"]]
#                         if locat_track != self.db_tracks:
#                             del_flag = True
#                     if del_flag:
#                         del locat_track[track["is_duplicate"]]
#                     self.database.update_idtrack(track["is_duplicate"], track["trk_id"], 0, self.camera)
#                     # #print("Track {} is a duplicate...".format(track["trk_id"]))
#                     continue

#                 if self.isExistTrkID(track["trk_id"],0) or self.isExistTrkID(track["trk_id"],1) or self.isExistTrkID(track["trk_id"],2):
#                     if self.isExistTrkID(track["trk_id"],1):
#                         pass
#                         #print("ID = {} Age = {} Hits = {}".format(track["trk_id"], track["age"], track["hits"]))
#                     continue

#                 if track["trk_id"] in self.duplicates:
#                     # print("Track {} in duplicates".format(track["trk_id"]))
#                     continue

#                 track["frame_ids"] = deque(np.asarray(track["frame_ids"]), maxlen=self.buffer_size)
#                 track["bboxes"] = deque(np.asarray(track["bboxes"]), maxlen=self.buffer_size)
#                 track["tlbr"] = np.asarray(track["tlbr"])
#                 track["mean"] = np.asarray(track["mean"])
#                 track["covariance"] = np.asarray(track["covariance"])
#                 track["sum"] = np.asarray(track["sum"])
#                 track["avg"] = np.asarray(track["avg"])
#                 track["last_feat"] = np.asarray(track["last_feat"])
#                 track["keypoints"] = np.asarray(track["keypoints"],np.float32)
#                 track["prev_keypoints"] = np.asarray(track["prev_keypoints"],np.float32)
#                 track["label"] = np.int64(track["label"])
#                 state = track["mean"], track["covariance"]
#                 # #print("SUM={} class={}".format(track["sum"],type(track["sum"])) )
#                 # #print("AVG={} class={}".format(track["avg"],type(track["avg"])) )
#                 # #print("Count={} class={}".format(track["count"],type(track["count"])) )

#                 db_track = Track(frame_id=track["frame_id"], trk_id=track["trk_id"],
#                 frame_ids=track["frame_ids"], bboxes=track["bboxes"], state=state, label=track["label"],
#                 tlbr=track["tlbr"], confirm_hits=track["confirm_hits"],age=track["age"], hits=track["hits"],
#                 _sum = track["sum"], avg=track["avg"], count=track["count"], last_feat=track["last_feat"],
#                 inlier_ratio=track["inlier_ratio"], keypoints=track["keypoints"], prev_keypoints=track["prev_keypoints"])

#                 self.db_tracks[db_track.trk_id] = db_track
#                 # #print("ADDED {} to DB tracks". format(db_track.trk_id))

#     def findTrk(self, trk_id):
#         track = None
#         location = None
#         try:
#             track = self.tracks[trk_id]
#             location = self.tracks
#         except (KeyError, IndexError):
#             try:
#                 track = self.db_tracks[trk_id]
#                 location = self.db_tracks
#             except (KeyError, IndexError):
#                 try:
#                     track = self.lost[trk_id]
#                     location = self.lost
#                 except (KeyError, IndexError):
#                     return track, location
#         finally:
#             return track, location

#     def isExistTrkID(self, trk_id, tracks=2):
#         if tracks == 0:
#             try:
#                 return self.tracks[trk_id] != None
#             except KeyError:
#                 return False
#         elif tracks == 1:
#             try:
#                 return self.db_tracks[trk_id] != None 
#             except KeyError:
#                 return False
#         else:
#             try:
#                 return self.lost[trk_id] != None
#             except KeyError:
#                 return False

#     def _group_tracks_by_depth(self, group_size=2):
#         n_depth = (self.max_age + group_size) // group_size
#         confirmed_by_depth = [[] for _ in range(n_depth)]
#         unconfirmed = []
#         # #print("n_depth={}, confirmed_by_depth={}".format(n_depth,confirmed_by_depth))
#         for trk_id, track in self.tracks.items():
#             if track.confirmed:
#                 depth = track.age // group_size
#                 # #print("age={}, depth={}".format(track.age,depth))
#                 confirmed_by_depth[depth].append(trk_id)
#             else:
#                 unconfirmed.append(trk_id)
#         return confirmed_by_depth, unconfirmed

#     def _matching_cost(self, trk_ids, detections, embeddings, occluded_dmask):
#         n_trk, n_det = len(trk_ids), len(detections)
#         if n_trk == 0 or n_det == 0:
#             return np.empty((n_trk, n_det))

#         features = np.empty((n_trk, embeddings.shape[1]))
#         invalid_fmask = np.zeros(n_trk, np.bool_)
#         for i, trk_id in enumerate(trk_ids):
#             track = self.tracks[trk_id]
#             if track.avg_feat.is_valid():
#                 features[i, :] = track.avg_feat()
#             else:
#                 invalid_fmask[i] = True

#         empty_mask = invalid_fmask[:, None] | occluded_dmask
#         fill_val = min(self.max_assoc_cost + 0.1, 1.)
#         cost = cdist(features, embeddings, self.metric, empty_mask, fill_val)

#         # fuse motion information
#         for row, trk_id in enumerate(trk_ids):
#             track = self.tracks[trk_id]
#             m_dist = self.kf.motion_distance(*track.state, detections.tlbr)
#             fuse_motion(cost[row], m_dist, self.motion_weight)

#         # make sure associated pair has the same class label
#         t_labels = np.fromiter((self.tracks[trk_id].label for trk_id in trk_ids), int, n_trk)
#         gate_cost(cost, t_labels, detections.label, self.max_assoc_cost)
#         return cost

#     def _iou_cost(self, trk_ids, detections):
#         n_trk, n_det = len(trk_ids), len(detections)
#         if n_trk == 0 or n_det == 0:
#             return np.empty((n_trk, n_det))

#         t_labels = np.fromiter((self.tracks[trk_id].label for trk_id in trk_ids), int, n_trk)
#         t_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids])
#         d_bboxes = detections.tlbr
#         iou_cost = iou_dist(t_bboxes, d_bboxes)
#         gate_cost(iou_cost, t_labels, detections.label, 1. - self.iou_thresh)
#         return iou_cost

#     def _reid_cost(self, dico_ids, detections, embeddings, dic_num=1, dicto=None):
#         if dic_num == 0:
#             dico = self.tracks
#         elif dic_num == 1:
#             dico = self.lost
#         elif dic_num == 2:
#             dico = self.db_tracks
#         else:
#             dico = dicto
#         n_hist, n_det = len(dico_ids), len(detections)
#         if n_hist == 0 or n_det == 0:
#             return np.empty((n_hist, n_det))

#         features = np.concatenate([dico[trk_id].avg_feat()
#                                    for trk_id in dico_ids]).reshape(n_hist, -1)
#         # if dic_num != 2:
#         #     for i in dico_ids:
#         #         #print("Dico meant to be",dico[i].avg_feat(), type(dico[i].avg_feat()))
#         #         break
#         #     #print("Features meant to be",features, type(features))
#         cost = cdist(features, embeddings, self.metric)

#         t_labels = np.fromiter((t.label for t in dico.values()), int, n_hist)
#         gate_cost(cost, t_labels, detections.label)
#         return cost

#     def _rectify_matches(self, matches, u_trk_ids, detections):
#         matches, u_trk_ids = set(matches), set(u_trk_ids)
#         #print("matches = {} u_trk_ids = {}".format(matches, u_trk_ids))
#         inactive_matches = [match for match in matches if not self.tracks[match[0]].active]

#         u_active = [trk_id for trk_id in u_trk_ids
#                     if self.tracks[trk_id].confirmed and self.tracks[trk_id].active]
#         #print("inactive_matches {} / u_active {}".format(inactive_matches, u_active))

#         n_inactive_matches = len(inactive_matches)
#         if n_inactive_matches == 0 or len(u_active) == 0:
#             return matches, u_trk_ids
#         m_inactive, det_ids = zip(*inactive_matches)
#         t_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in u_active])
#         d_bboxes = detections[det_ids,].tlbr
#         iou_cost = iou_dist(t_bboxes, d_bboxes)

#         col_indices = list(range(n_inactive_matches))
#         dup_matches, _, _ = greedy_match(iou_cost, u_active, col_indices,
#                                          1. - self.duplicate_thresh, 1, "Dupli-Match")
#         #print("dup_matches = {}".format(dup_matches))
#         for u_trk_id, col in dup_matches:
#             m_trk_id, det_id = m_inactive[col], det_ids[col]
#             t_u_active, t_m_inactive = self.tracks[u_trk_id], self.tracks[m_trk_id]
#             if t_m_inactive.end_frame < t_u_active.start_frame:
#                 LOGGER.debug(f"{'Merged:':<14}{u_trk_id} -> {m_trk_id}")
#                 t_m_inactive.merge_continuation(t_u_active)
#                 u_trk_ids.remove(u_trk_id)
#                 del self.tracks[u_trk_id]
#             else:
#                 LOGGER.debug(f"{'Duplicate:':<14}{m_trk_id} -> {u_trk_id}")
#                 u_trk_ids.remove(u_trk_id)
#                 u_trk_ids.add(m_trk_id)
#                 matches.remove((m_trk_id, det_id))
#                 matches.add((u_trk_id, det_id))
#         return list(matches), list(u_trk_ids)

#     def _remove_duplicate(self, trk_ids1, trk_ids2):
#         if len(trk_ids1) == 0 or len(trk_ids2) == 0:
#             return

#         bboxes1 = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids1])
#         bboxes2 = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids2])

#         ious = bbox_ious(bboxes1, bboxes2)
#         idx = np.where(ious >= self.duplicate_thresh)
#         dup_ids = set()
#         for row, col in zip(*idx):
#             trk_id1, trk_id2 = trk_ids1[row], trk_ids2[col]
#             track1, track2 = self.tracks[trk_id1], self.tracks[trk_id2]
#             if len(track1) > len(track2):
#                 dup_ids.add(trk_id2)
#             else:
#                 dup_ids.add(trk_id1)
#         for trk_id in dup_ids:
#             LOGGER.debug(f"{'Duplicate:':<14}{self.tracks[trk_id]}")
#             del self.tracks[trk_id]
