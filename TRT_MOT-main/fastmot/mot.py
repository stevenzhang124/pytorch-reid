from enum import Enum
import logging
import time
import cv2

from .detector import SSDDetector, YoloDetector, PublicDetector
from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils.visualization import draw_tracks, draw_detections
from .utils.visualization import draw_flow_bboxes, draw_background_flow



LOGGER = logging.getLogger(__name__)


class DetectorType(Enum):
    SSD = 0
    YOLO = 1
    PUBLIC = 2


class MOT:
    """
    This is the top level module that integrates detection, feature extraction,
    and tracking together.
    Parameters
    ----------
    size : (int, int)
        Width and height of each frame.
    capture_dt : float
        Time interval in seconds between each captured frame.
    config : Dict
        Tracker configuration.
    draw : bool
        Flag to toggle visualization drawing.
    verbose : bool
        Flag to toggle output verbosity.
    """

    def __init__(self, size, capture_dt, config, camera, database, fps, draw=False, verbose=False, cuda_ctx=None):
        self.size = size
        self.draw = draw
        self.verbose = verbose
        self.detector_type = DetectorType[config['detector_type']]
        self.detector_frame_skip = config['detector_frame_skip']
        self.cuda_ctx = cuda_ctx

        LOGGER.info('Loading detector model...')
        if self.detector_type == DetectorType.SSD:
            self.detector = SSDDetector(self.size, config['ssd_detector'],self.cuda_ctx)
        elif self.detector_type == DetectorType.YOLO:
            self.detector = YoloDetector(self.size, config['yolo_detector'],self.cuda_ctx)
        elif self.detector_type == DetectorType.PUBLIC:
            self.detector = PublicDetector(self.size, config['public_detector'],self.cuda_ctx)

        LOGGER.info('Loading feature extractor model...')
        self.extractor = FeatureExtractor(config['feature_extractor'],self.cuda_ctx)
        self.tracker = MultiTracker(self.size, capture_dt, self.extractor.metric,
                                    config['multi_tracker'],camera, database)

        # reset counters
        self.frame_count = 0
        self.detector_frame_count = 0
        self.preproc_time = 0
        self.detector_time = 0
        self.extractor_time = 0
        self.association_time = 0
        self.tracker_time = 0
        self.fps_time = 0
        self.fps = fps

    @property
    def visible_tracks(self):
        # retrieve confirmed and active tracks from the tracker
        return [track for track in self.tracker.tracks.values()
                if track.confirmed and track.active]

    def initiate(self):
        """
        Resets multiple object tracker.
        """
        self.frame_count = 0

    def step(self, frame):
        """
        Runs multiple object tracker on the next frame.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        detections = []
        fps_time = time.perf_counter()
        if self.cuda_ctx:
            self.cuda_ctx.push()

        if self.frame_count == 0:
            detections = self.detector(self.frame_count, frame)
            self.tracker.initiate(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                tic = time.perf_counter()
                self.detector.detect_async(self.frame_count, frame)             # Detect
                self.preproc_time += time.perf_counter() - tic

                # Compute optical flow
                tic = time.perf_counter()
                self.tracker.compute_flow(frame)                                # Compute Flow
                detections = self.detector.postprocess()
                self.detector_time += time.perf_counter() - tic

                # Apply Kalman Filter
                tic = time.perf_counter()
                self.extractor.extract_async(frame, detections)
                self.tracker.apply_kalman()                                     # Apply Kalman
                embeddings = self.extractor.postprocess()
                self.extractor_time += time.perf_counter() - tic

                # Update tracks with database information and next_id to match database length
                self.tracker._update_tracks()

                tic = time.perf_counter()
                self.tracker.update(self.frame_count, detections, embeddings)   # Update
                self.association_time += time.perf_counter() - tic
                self.detector_frame_count += 1
            else:
                tic = time.perf_counter()
                self.tracker.track(frame)                                       # Track -> Flow + Apply Kalman 
                self.tracker_time += time.perf_counter() - tic
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        if self.draw:
            self._draw(frame, detections)
        frame_rate_calc = min(self.fps,1 / (time.perf_counter() - fps_time))
        # frame_rate_calc = round(self.frame_count / max(1,elapsed_time))
        cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)
        if self.frame_count % 100 == 0:
            print("Local tracks keys = ",self.tracker.tracks.keys())
            print("Local lost keys = ",self.tracker.lost.keys())
            print("Local db keys = ",self.tracker.db_tracks.keys())
        self.frame_count += 1


    def _draw(self, frame, detections):
        draw_tracks(frame, self.visible_tracks, show_flow=self.verbose)
        if self.verbose:
            draw_detections(frame, detections)
            draw_flow_bboxes(frame, self.tracker)
            draw_background_flow(frame, self.tracker)
        cv2.putText(frame, f'Visible: {len(self.visible_tracks)}', (20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2, cv2.LINE_AA)

    def getAverages(self, elapsed_time, logger):
        # timing results
        avg_fps = round(self.frame_count / max(1,elapsed_time))
        avg_tracker_time = self.tracker_time / max(1, (self.frame_count - self.detector_frame_count))
        avg_extractor_time = self.extractor_time / max(1, self.detector_frame_count)
        avg_preproc_time = self.preproc_time / max(1, self.detector_frame_count)
        avg_detector_time = self.detector_time / max(1, self.detector_frame_count)
        avg_assoc_time = self.association_time / max(1, self.detector_frame_count)

        logger.info('Average FPS: %d', avg_fps)
        logger.info('Average tracker time: %f', avg_tracker_time)
        logger.info('Average feature extractor time: %f', avg_extractor_time)
        logger.info('Average preprocessing time: %f', avg_preproc_time)
        logger.info('Average detector time: %f', avg_detector_time)
        logger.info('Average association time: %f', avg_assoc_time)
        