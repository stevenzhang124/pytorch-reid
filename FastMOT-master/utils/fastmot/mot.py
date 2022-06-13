from types import SimpleNamespace
from enum import Enum
import logging
import cv2
import time

from ..detector import SSDDetector, YOLODetector, PublicDetector
from ..feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .. import Profiler
from ..visualization import Visualizer



LOGGER = logging.getLogger(__name__)


class DetectorType(Enum):
    SSD = 0
    YOLO = 1
    PUBLIC = 2


class MOT:
    def __init__(self, size, camera, database, fps, par,
                 detector_type='YOLO',
                 detector_frame_skip=5,
                 ssd_detector_cfg=None,
                 yolo_detector_cfg=None,
                 public_detector_cfg=None,
                 feature_extractor_cfg=None,
                 tracker_cfg=None,
                 visualizer_cfg=None,
                 draw=False,
                 verbose=False,
                 cuda_ctx=None):
        """Top level module that integrates detection, feature extraction,
        and tracking together.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        detector_type : {'SSD', 'YOLO', 'public'}, optional
            Type of detector to use.
        detector_frame_skip : int, optional
            Number of frames to skip for the detector.
        ssd_detector_cfg : SimpleNamespace, optional
            SSD detector configuration.
        yolo_detector_cfg : SimpleNamespace, optional
            YOLO detector configuration.
        public_detector_cfg : SimpleNamespace, optional
            Public detector configuration.
        feature_extractor_cfg : SimpleNamespace, optional
            Feature extractor configuration.
        tracker_cfg : SimpleNamespace, optional
            Tracker configuration.
        visualizer_cfg : SimpleNamespace, optional
            Visualization configuration.
        draw : bool, optional
            Enable visualization.
        """
        self.size = size
        self.detector_type = DetectorType[detector_type.upper()]
        assert detector_frame_skip >= 1
        self.detector_frame_skip = detector_frame_skip
        self.draw = draw
        self.cuda_ctx = cuda_ctx

        if ssd_detector_cfg is None:
            ssd_detector_cfg = SimpleNamespace()
        if yolo_detector_cfg is None:
            yolo_detector_cfg = SimpleNamespace()
        if public_detector_cfg is None:
            public_detector_cfg = SimpleNamespace()
        if feature_extractor_cfg is None:
            feature_extractor_cfg = SimpleNamespace()
        if tracker_cfg is None:
            tracker_cfg = SimpleNamespace()
        if visualizer_cfg is None:
            visualizer_cfg = SimpleNamespace()

        LOGGER.info('Loading detector model...')
        if self.detector_type == DetectorType.SSD:
            self.detector = SSDDetector(self.size, **vars(ssd_detector_cfg), cuda_ctx=cuda_ctx)
        elif self.detector_type == DetectorType.YOLO:
            self.detector = YOLODetector(self.size, **vars(yolo_detector_cfg), cuda_ctx=cuda_ctx)
        elif self.detector_type == DetectorType.PUBLIC:
            self.detector = PublicDetector(self.size, self.detector_frame_skip,
                                           **vars(public_detector_cfg), cuda_ctx=cuda_ctx)

        LOGGER.info('Loading feature extractor model...')
        self.extractor = FeatureExtractor(**vars(feature_extractor_cfg))
        self.tracker = MultiTracker(self.size, self.extractor.metric, camera, database, par, **vars(tracker_cfg))

        self.visualizer = Visualizer(**vars(visualizer_cfg),verbose=verbose)
        self.frame_count = 0
        self.fps = fps

    def visible_tracks(self):
        """Retrieve visible tracks from the tracker

        Returns
        -------
        Iterator[Track]
            Confirmed and active tracks from the tracker.
        """
        return (track for track in self.tracker.tracks.values()
                if track.confirmed and track.active)

    def reset(self, cap_dt):
        """Resets multiple object tracker. Must be called before `step`.

        Parameters
        ----------
        cap_dt : float
            Time interval in seconds between each frame.
        """
        self.frame_count = 0
        self.tracker.reset(cap_dt)

    def step(self, frame):
        """Runs multiple object tracker on the next frame.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        detections = []
        fps_time = time.perf_counter()
        frame_rate_calc = 0

        if self.cuda_ctx:
            self.cuda_ctx.push()

        if self.frame_count == 0:
            detections = self.detector(self.frame_count, frame)
            self.tracker.init(frame, detections)
        else:
            # if self.frame_count % 3 != 0:
            #     self.frame_count += 1
            #     return

            if self.frame_count % self.detector_frame_skip == 0:
                with Profiler('preproc'):   
                    self.detector.detect_async(self.frame_count, frame)             # Detect

                with Profiler('detect'):
                    with Profiler('track'):
                        self.tracker.compute_flow(frame)                            # Compute Flow
                    detections = self.detector.postprocess()

                with Profiler('extract'):
                    self.extractor.extract_async(frame, detections.tlbr)
                    with Profiler('track', aggregate=True):
                        self.tracker.apply_kalman(frame)                            # Apply Kalman
                    embeddings = self.extractor.postprocess()
                
                with Profiler('refresh'):
                    # pass
                    self.tracker._update_tracks()

                # with Profiler('merger'):
                #     self.tracker._secmerge_tracks(1)
                #     self.tracker._secmerge_tracks(2)

                with Profiler('assoc'):
                    self.tracker.update(frame, self.frame_count, detections, embeddings)   # Update

            else:
                with Profiler('track'):
                    self.tracker.track(frame)                                       # Track -> Flow + Apply Kalman 

        if self.cuda_ctx:
            self.cuda_ctx.pop()

        if self.draw:
            self._draw(frame, detections)
        frame_rate_calc = min(self.fps,1 / (time.perf_counter() - fps_time))
        # frame_rate_calc = round(self.detector_frame_skip / (time.perf_counter() - fps_time))
        # frame_rate_calc = round(self.frame_count / max(1,elapsed_time))
        # colr = (255, 87, 51 )
        colr = (255, 255, 255 )
        # cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
        #             cv2.FONT_HERSHEY_PLAIN, 1, colr, 2, cv2.LINE_AA)
        # if self.frame_count % 100 == 0:
        #     print("Local tracks keys = ",self.tracker.tracks.keys())
        #     print("Local lost keys = ",self.tracker.lost.keys())
        #     print("Local db keys = ",self.tracker.db_tracks.keys())
        self.frame_count += 1

    def _draw(self, frame, detections):
        visible_tracks = list(self.visible_tracks())
        self.visualizer.render(frame, visible_tracks, detections, self.tracker.klt_bboxes.values(),
                               self.tracker.flow.prev_bg_keypoints, self.tracker.flow.bg_keypoints,
                               self.frame_count, self.tracker.camera)
        # cv2.putText(frame, f'Visible: {len(visible_tracks)}', (30, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
        # if self.tracker.camera == '117':
        cap_room = 20
        # print("ELCou",self.tracker.person_count)
        self.tracker.entries = self.tracker.database.get_entry() if self.tracker.database else self.tracker.entries
        self.tracker.exits = self.tracker.database.get_exit() if self.tracker.database else self.tracker.exits
        # print("2nd -> ELCou",self.tracker.person_count)
        # occ_room = self.tracker.person_count/ cap_room
        occ_room = len(visible_tracks) / cap_room
        # if self.tracker.camera == '117':
        # cv2.putText(frame, f'VISIBLE: {len(visible_tracks)}', (5, 20),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        if self.tracker.camera == '113' or self.tracker.camera == '106' or self.tracker.camera == '115':
            text = "VISIBLE:\n{}/{}\n{} %\nIN: {}\nOUT: {}".format(len(visible_tracks), cap_room, round(occ_room*100), self.tracker.entries, self.tracker.exits)
            y0, dy = 20, 25
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                if i < 3:
                    cv2.putText(frame, line, (500, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int((1-occ_room)*255), int((1-occ_room)*255), 255), 1, cv2.LINE_AA)
                else:
                    color = (255, 255, 255)
                    color = (255, 255, 0)
                    cv2.putText(frame, line, (500, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        else:
            text = "VISIBLE:\n{}/{}\n{} %".format(len(visible_tracks), cap_room, round(occ_room*100))
            y0, dy = 20, 25
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                cv2.putText(frame, line, (500, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int((1-occ_room)*255), int((1-occ_room)*255), 255), 1, cv2.LINE_AA)
        

    def getAverages(self, avg_fps, logger):
        # timing results
        # avg_fps = round(self.frame_count / max(1,elapsed_time))
        # avg_fps = min(self.fps,1 / (time.perf_counter() - elapsed_time))
        LOGGER.debug('=================Timing Stats=================')
        logger.info('Average FPS: %d', avg_fps)
        LOGGER.info(f"{'Average tracker time:':<30}{Profiler.get_avg_millis('track'):>6.3f} ms")
        LOGGER.info(f"{'Average preprocess time:':<30}{Profiler.get_avg_millis('preproc'):>6.3f} ms")
        LOGGER.info(f"{'Average detector/flow time:':<30}{Profiler.get_avg_millis('detect'):>6.3f} ms")
        LOGGER.info(f"{'Average feature extracter time:':<30}"
                     f"{Profiler.get_avg_millis('extract'):>6.3f} ms")
        LOGGER.info(f"{'Average association time:':<30}{Profiler.get_avg_millis('assoc'):>6.3f} ms")
        LOGGER.info(f"{'Average database update time:':<30}{Profiler.get_avg_millis('refresh'):>6.3f} ms")
        LOGGER.info(f"{'Average merge tracks time:':<30}{Profiler.get_avg_millis('merger'):>6.3f} ms")