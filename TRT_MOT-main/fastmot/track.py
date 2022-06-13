import numpy as np

from .models import LABEL_MAP
from .utils.rect import get_center

import json
import base64


class Track:
    def __init__(self, frame_id, trk_id, tlbr, state, label,
     age=0, hits=0, alpha=0, smooth_feature=None, inlier_ratio=1.,
     keypoints=np.empty((0, 2), np.float32), prev_keypoints=np.empty((0, 2), np.float32)):
        self.start_frame = frame_id
        self.trk_id = trk_id
        self.tlbr = tlbr
        self.state = state
        self.label = label

        self.age = age
        self.hits = hits
        self.alpha = alpha
        self.smooth_feature = smooth_feature

        self.inlier_ratio = inlier_ratio
        self.keypoints = keypoints
        self.prev_keypoints = prev_keypoints

    def __str__(self):
        return "%s %d at %s" % (LABEL_MAP[self.label], self.trk_id,
                                get_center(self.tlbr).astype(int))

    def __repr__(self):
        # return self.__str__()
        return "start_frame=%s trk_id=%s tlbr=%s\nstate=%s\nlabel=%s\n"
        "age=%s hits=%s alpha=%s smooth_feature=%s\n"
        "inlier_ratio=%s keypoints=%s prev_keypoints=%s" % (self.start_frame, self.trk_id, (self.tlbr).astype(int), 
        self.state, self.label, self.age, self.hits, self.alpha, self.smooth_feature,
        self.inlier_ratio, self.keypoints, self.prev_keypoints )

    def __lt__(self, other):
        # ordered by approximate distance to the image plane, closer is greater
        return (self.tlbr[-1], -self.age) < (other.tlbr[-1], -other.age)

    def _to_dict(self):
        
        return {
            "frame_id":self.start_frame,
            "trk_id": self.trk_id,
            "tlbr": self.tlbr.tolist(),
            # "state": self.state,
            "mean": np.array(self.state[0]).tolist(), #np.array(self.state).tolist()
            "covariance" : np.array(self.state[1]).tolist(),
            "label": int(self.label),
            
            "age": self.age,
            "hits":self.hits,
            "alpha": self.alpha,
            "smooth_feature": self.smooth_feature.tolist(),
            
            "inlier_ratio": self.inlier_ratio,
            "keypoints": self.keypoints.tolist(),
            "prev_keypoints": self.prev_keypoints.tolist()
        }


    @property
    def active(self):
        return self.age < 2

    @property
    def confirmed(self):
        return self.hits > 0

    def update(self, tlbr, state, embedding=None):
        self.tlbr = tlbr
        self.state = state
        if embedding is not None:
            self.age = 0
            self.hits += 1
            self.update_feature(embedding)

    def reactivate(self, frame_id, tlbr, state, embedding):
        self.start_frame = frame_id
        self.tlbr = tlbr
        self.state = state
        self.age = 0
        self.update_feature(embedding)
        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)

    def mark_missed(self):
        self.age += 1

    def update_feature(self, embedding):
        if self.smooth_feature is None:
            self.smooth_feature = embedding
        else:
            self.smooth_feature = self.alpha * self.smooth_feature + (1. - self.alpha) * embedding
            self.smooth_feature /= np.linalg.norm(self.smooth_feature)

# main type <class 'dict'>
# dtype =  frame_id <class 'int'>
# dtype =  trk_id <class 'int'>
# dtype =  tlbr <class 'list'>
# dtype =  mean <class 'list'>
# dtype =  covariance <class 'list'>
# dtype =  label <class 'int'>
# dtype =  age <class 'int'>
# dtype =  hits <class 'int'>
# dtype =  alpha <class 'int'>
# dtype =  smooth_feature <class 'list'>
# dtype =  inlier_ratio <class 'float'>
# dtype =  keypoints <class 'list'>
# dtype =  prev_keypoints <class 'list'>


# main type <class 'dict'>
# dtype =  frame_id <class 'int'>
# dtype =  trk_id <class 'int'>
# dtype =  tlbr <class 'numpy.ndarray'>
# dtype =  mean <class 'numpy.ndarray'>
# dtype =  covariance <class 'numpy.ndarray'>
# dtype =  label <class 'numpy.int64'>
# dtype =  age <class 'int'>
# dtype =  hits <class 'int'>
# dtype =  alpha <class 'int'>
# dtype =  smooth_feature <class 'numpy.ndarray'>
# dtype =  inlier_ratio <class 'float'>
# dtype =  keypoints <class 'numpy.ndarray'>
# dtype =  prev_keypoints <class 'numpy.ndarray'>