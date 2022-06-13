from collections import deque
import numpy as np
import numba as nb

from .fastmot.models import get_label_name
from .distance import cdist, cosine
from .numba import apply_along_axis, normalize_vec
from .rect import get_center

# from .models import get_label_name
# from .utils.distance import cdist, cosine
# from .utils.numba import apply_along_axis, normalize_vec
# from .utils.rect import get_center


class ClusterFeature:
    def __init__(self, num_clusters, metric):
        self.num_clusters = num_clusters
        self.metric = metric
        self.clusters = None
        self.cluster_sizes = None
        self._next_idx = 0

    def __len__(self):
        return self._next_idx

    def __call__(self):
        return self.clusters[:self._next_idx]

    def update(self, embedding):
        if self._next_idx < self.num_clusters:
            if self.clusters is None:
                self.clusters = np.empty((self.num_clusters, len(embedding)), embedding.dtype)
                self.cluster_sizes = np.zeros(self.num_clusters, int)
            self.clusters[self._next_idx] = embedding
            self.cluster_sizes[self._next_idx] += 1
            self._next_idx += 1
        else:
            nearest_idx = self._get_nearest_cluster(self.clusters, embedding)
            self.cluster_sizes[nearest_idx] += 1
            self._seq_kmeans(self.clusters, self.cluster_sizes, embedding, nearest_idx)

    def distance(self, embeddings):
        if self.clusters is None:
            return np.ones(len(embeddings))
        clusters = normalize_vec(self.clusters[:self._next_idx])
        return apply_along_axis(np.min, cdist(clusters, embeddings, self.metric), axis=0)

    def merge(self, features, other, other_features):
        if len(features) > len(other_features):
            for feature in other_features:
                if feature is not None:
                    self.update(feature)
        else:
            for feature in features:
                if feature is not None:
                    other.update(feature)
            self.clusters = other.clusters.copy()
            self.clusters_sizes = other.cluster_sizes.copy()
            self._next_idx = other._next_idx

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_nearest_cluster(clusters, embedding):
        return np.argmin(cosine(np.atleast_2d(embedding), clusters))

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _seq_kmeans(clusters, cluster_sizes, embedding, idx):
        div_size = 1. / cluster_sizes[idx]
        clusters[idx] += (embedding - clusters[idx]) * div_size


class SmoothFeature:
    def __init__(self, learning_rate):
        self.lr = learning_rate
        self.smooth = None

    def __call__(self):
        return self.smooth

    def update(self, embedding):
        if self.smooth is None:
            self.smooth = embedding.copy()
        else:
            self._rolling(self.smooth, embedding, self.lr)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _rolling(smooth, embedding, lr):
        smooth[:] = (1. - lr) * smooth + lr * embedding
        norm_factor = 1. / np.linalg.norm(smooth)
        smooth *= norm_factor


class AverageFeature:
    def __init__(self,_sum=None, avg=None, count=0):
        self.sum = _sum
        self.avg = avg
        self.count = count

    def __call__(self):
        return self.avg

    def is_valid(self):
        return self.count > 0

    def update(self, embedding):
        self.count += 1
        if self.sum is None:
            self.sum = embedding.copy()
            self.avg = embedding.copy()
        else:
            self._average(self.sum, self.avg, embedding, self.count)

    def merge(self, other):
        self.count += other.count
        if self.sum is None:
            self.sum = other.sum
            self.avg = other.avg
        elif other.sum is not None:
            self._average(self.sum, self.avg, other.sum, self.count)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _average(_sum, avg, embed, count):
        _sum += embed
        div_cnt = 1. / count
        avg[:] = _sum * div_cnt
        norm_factor = 1. / np.linalg.norm(avg)
        avg *= norm_factor


class Track:
    _count = 0

    def __init__(self, frame_id, trk_id, tlbr, state, label, confirm_hits=1, frame_ids=None, bboxes=None,  buffer_size=30,
         age=0, hits=0, _sum=None, avg=None, count=0, last_feat=None, inlier_ratio=1.,
     keypoints=np.empty((0, 2), np.float32), prev_keypoints=np.empty((0, 2), np.float32),direction=0):
        self.start_frame = frame_id
        self.trk_id =  trk_id
        self.frame_ids = frame_ids if frame_ids is not None else deque([frame_id], maxlen=buffer_size)
        self.bboxes = bboxes if bboxes is not None else deque([tlbr], maxlen=buffer_size)
        self.confirm_hits = confirm_hits
        self.state = state
        self.label = label

        self.age = age
        self.hits = hits
        self.avg_feat = AverageFeature(_sum, avg, count)
        self.last_feat = last_feat

        self.inlier_ratio = inlier_ratio
        self.keypoints = keypoints
        self.prev_keypoints = prev_keypoints
        self.direction = direction

    def __str__(self):
        # print("TLBR",self.tlbr)
        # print("BBOXES",self.bboxes)

        x, y = get_center(self.tlbr)
        return f'{get_label_name(self.label)} {self.trk_id:>3} at ({int(x):>4}, {int(y):>3})'

    def __repr__(self):
        return self.__str__()
        # return "start_frame=%s trk_id=%s tlbr=%s\nstate=%s\nlabel=%s\n"
        # "age=%s hits=%s avg_feat=%s last_feat=%s\n"
        # "inlier_ratio=%s keypoints=%s prev_keypoints=%s" % (self.start_frame, self.trk_id, (self.tlbr).astype(int), 
        # self.state, self.label, self.age, self.hits, self.avg_feat, self.last_feat,
        # self.inlier_ratio, self.keypoints, self.prev_keypoints )

    def __len__(self):
        return self.end_frame - self.start_frame

    def __lt__(self, other):
        # ordered by approximate distance to the image plane, closer is greater
        return (self.tlbr[-1], -self.age) < (other.tlbr[-1], -other.age)

    def _to_dict(self):
        return {
            "frame_id":self.start_frame,
            "trk_id": self.trk_id,
            "frame_ids": np.array(self.frame_ids).tolist(),
            "bboxes": np.array(self.bboxes).tolist(),
            "confirm_hits": self.confirm_hits,
            "tlbr": self.tlbr.tolist(),
            # "state": self.state,
            "mean": np.array(self.state[0]).tolist(), #np.array(self.state).tolist()
            "covariance" : np.array(self.state[1]).tolist(),
            "label": int(self.label),
            
            "age": self.age,
            "hits":self.hits,
            "sum": self.avg_feat.sum.tolist() if self.avg_feat.sum is not None else [],       #avg_feat
            "avg": self.avg_feat.avg.tolist() if self.avg_feat.avg is not None else [],       #avg_feat
            "count": self.avg_feat.count,           #avg_feat
            "last_feat": self.last_feat.tolist() if self.last_feat is not None else [],
            
            "inlier_ratio": self.inlier_ratio,
            "keypoints": self.keypoints.tolist(),
            "prev_keypoints": self.prev_keypoints.tolist()
        }

    @property
    def tlbr(self):
        return self.bboxes[-1]

    @property
    def end_frame(self):
        return self.frame_ids[-1]

    @property
    def active(self):
        return self.age < 2

    @property
    def confirmed(self):
        return self.hits >= self.confirm_hits

    def update(self, tlbr, state):
        self.bboxes.append(tlbr)
        self.state = state

    def add_detection(self, frame_id, tlbr, state, embedding, is_valid=True):
        self.frame_ids.append(frame_id)
        self.bboxes.append(tlbr)
        self.state = state
        if is_valid:
            self.last_feat = embedding
            self.avg_feat.update(embedding)
        self.age = 0
        self.hits += 1

    def reinstate(self, frame_id, tlbr, state, embedding):
        self.start_frame = frame_id
        self.frame_ids.append(frame_id)
        self.bboxes.append(tlbr)
        self.state = state
        self.last_feat = embedding
        self.avg_feat.update(embedding)
        self.age = 0
        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)

    def mark_missed(self):
        self.age += 1

    def merge_continuation(self, other, flag=True):
        if flag:
            self.avg_feat.merge(other.avg_feat)
            self.trk_id = min(self.trk_id, other.trk_id)
            return
        self.frame_ids.extend(other.frame_ids)
        self.bboxes.extend(other.bboxes)
        self.state = other.state
        self.age = other.age
        self.hits += other.hits

        self.keypoints = other.keypoints
        self.prev_keypoints = other.prev_keypoints

        if other.last_feat is not None:
            self.last_feat = other.last_feat
        self.avg_feat.merge(other.avg_feat)
        self.trk_id = min(self.trk_id, other.trk_id)

    @staticmethod
    def next_id():
        Track._count += 1
        return Track._count

# start_frame <class 'int'>
# trk_id <class 'int'>
# frame_ids <class 'collections.deque'>
# bboxes <class 'collections.deque'>
# confirm_hits <class 'int'>
# label <class 'numpy.int64'>
# age <class 'int'>
# hits <class 'int'>
# sum <class 'numpy.ndarray'>
# avg <class 'numpy.ndarray'>
# count <class 'int'>
# last_feat <class 'numpy.ndarray'>
# inlier_ratio <class 'float'>
# keypoints <class 'numpy.ndarray'>
# prev_keypoints <class 'numpy.ndarray'>


# frame_id --> <class 'int'>
# trk_id --> <class 'int'>
# frame_ids --> <class 'list'>
# bboxes --> <class 'list'>
# confirm_hits --> <class 'int'>
# mean --> <class 'list'>
# covariance --> <class 'list'>
# label --> <class 'int'>
# age --> <class 'int'>
# hits --> <class 'int'>
# sum --> <class 'list'>
# avg --> <class 'list'>
# count --> <class 'int'>
# last_feat --> <class 'list'>
# inlier_ratio --> <class 'float'>
# keypoints --> <class 'list'>
# prev_keypoints --> <class 'list'>
# trk_id -->  <class 'int'>

# print("\nstart_frame",type(self.start_frame))
# print("trk_id",type(self.trk_id))
# print("frame_ids",type(self.frame_ids))
# print("bboxes",type(self.bboxes))
# print("confirm_hits",type(self.confirm_hits))
# print("label",type(self.label))
# print("age",type(self.age))
# print("hits",type(self.hits))
# print("sum",type(self.avg_feat.sum))
# print("avg",type(self.avg_feat.avg))
# print("count",type(self.avg_feat.count))
# print("last_feat",type(self.last_feat))
# print("inlier_ratio",type(self.inlier_ratio))
# print("keypoints",type(self.keypoints))
# print("prev_keypoints",type(self.prev_keypoints))