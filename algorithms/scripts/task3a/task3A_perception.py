#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 3A of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          6494
# Author List:      Krishang Nigam, Harshil Jayswal, Masum Pancholi, Divy Vaghasiya
# Filename:         task3A_perception.py
# Functions:
#                   main
# Nodes:            Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw,
#                                          /camera/camera/aligned_depth_to_color/image_raw,
#                                          /camera/camera/color/camera_info, /ore_picked,
#                                          /ebot_at_arm ]


################### IMPORT MODULES #######################

from collections import deque
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_geometry_msgs import do_transform_point
import tf2_ros


##################### TASK CONSTANTS #######################

# The three kinds of ore; each detected ore is published as <kind>_<n>, n counting from 1.
ORE_TYPES = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

# The RealSense topics. The depth image is aligned to the colour image.
COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/camera/color/camera_info'
BASE_FRAME = 'base_link'                # parent frame of every ore transform

# Part of the image searched for ores (the conveyor), as fractions of width and height.
ROI_X_FRAC = 0.50                       # only the right half of the image
ROI_Y_MIN_FRAC = 0.40
ROI_Y_MAX_FRAC = 0.99

# A detection is accepted only if its 3D position (base_link, metres) lies in this box.
# The z band is the height of the ore top faces on the conveyor; x and y are kept wide.
CONVEYOR_BELT_X_MIN = -2.0
CONVEYOR_BELT_X_MAX = 1.0
CONVEYOR_BELT_Y_MIN = -2.0
CONVEYOR_BELT_Y_MAX = 1.0
CONVEYOR_BELT_Z_MIN = -0.085
CONVEYOR_BELT_Z_MAX = -0.045

# HSV thresholds (lower, upper) for the coloured top face of each kind of ore.
COLOR_BOUNDS = {
    'azurite_ore': (np.array([100, 180, 150], dtype=np.uint8),
                    np.array([112, 255, 255], dtype=np.uint8)),
    'malachite_ore': (np.array([50, 140, 50], dtype=np.uint8),
                      np.array([90, 255, 255], dtype=np.uint8)),
    'vanadinite_ore': (np.array([5, 120, 120], dtype=np.uint8),
                       np.array([16, 255, 255], dtype=np.uint8)),
}

# Shape checks on each colour blob.
MIN_AREA_BY_TYPE = {'azurite_ore': 20, 'malachite_ore': 20, 'vanadinite_ore': 15}   # pixels
MAX_CONTOUR_AREA = 100000               # pixels: upper bound on a single ore's blob
MAX_ASPECT_RATIO = 2.5                  # partly hidden ores are not square
MIN_FILL_RATIO = 0.40                   # blob area / area of its bounding rectangle

# Tracking and smoothing
TRACK_GATE_M = 0.04                     # nearest-neighbour gating distance in base_link (m)
CONFIRM_HITS = 2                        # detections before a track's transform is published
MEDIAN_FILTER_WINDOW = 5                # rolling history length for median smoothing
LIFT_OFF_Z_THRESHOLD = 0.04             # m above CONVEYOR_BELT_Z_MAX at which a track counts as picked
PROMOTE_HITS = 3                        # consecutive sightings before a new ore gets a name

# The ore package, on the eBot's top deck while the eBot stands beside the arm. Its top face
# carries a magenta patch (0.141 x 0.105 m, centred on the face); the transform is that centre.
PACKAGE_FRAME = 'ore_package'
PACKAGE_COLOR_BOUNDS = (np.array([135, 80, 30], dtype=np.uint8),     # magenta: hue ~150-170
                        np.array([179, 255, 255], dtype=np.uint8))
PACKAGE_MIN_AREA = 40                   # pixels (the patch is ~30 x 20 px at 1.9 m)
PACKAGE_MAX_ASPECT_RATIO = 3.0
PACKAGE_MIN_FILL_RATIO = 0.50
# Accepted region in base_link (m): around the arm stop pose, at the deck-top height. The
# deck top is ~0.18 m above the arm base and the package top ~0.25 m.
PACKAGE_X_MIN, PACKAGE_X_MAX = 0.25, 1.00
PACKAGE_Y_MIN, PACKAGE_Y_MAX = -0.50, 0.50
PACKAGE_Z_MIN, PACKAGE_Z_MAX = 0.15, 0.40
PACKAGE_CONFIRM_HITS = 3                # detections before the package transform is published

# Processing rates and debugging
IMAGE_TIMER_PERIOD = 0.10               # s: perception loop (10 Hz)
BROADCAST_TIMER_PERIOD = 0.05           # s: transform publishing (20 Hz)
DEBUG_LOG = True                        # once-per-second debug logging


##################### CLASS DEFINITION #######################

class ore_perception(Node):
    '''
    ___CLASS___

    Description:    Class which detects the ores on the conveyor in the RealSense images,
                    gives each one a stable name, and broadcasts its position as a
                    base_link -> <kind>_<n> transform until the ore is picked. Once the eBot
                    stands beside the arm, it also detects the ore package on the eBot's
                    deck and broadcasts it as base_link -> ore_package until it is picked.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ore_perception
        '''

        super().__init__(
            'task3A_perception_node',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        self.bridge = CvBridge()                                    # ROS Image <-> OpenCV conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                    # holds the camera -> base_link transform
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.broadcaster = tf2_ros.TransformBroadcaster(self)       # publishes the ore transforms

        ############ Topic SUBSCRIPTIONS ############

        self.color_sub = self.create_subscription(Image, COLOR_TOPIC, self.colorimagecb, 10)
        self.depth_sub = self.create_subscription(Image, DEPTH_TOPIC, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.caminfocb, 10)
        self.picked_sub = self.create_subscription(String, '/ore_picked', self.ore_picked_cb, 10)

        # /ebot_at_arm is latched by the planner, so subscribe with the same durability
        latched = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.at_arm_sub = self.create_subscription(Bool, '/ebot_at_arm', self.ebotatarmcb, latched)

        ############ Constructor VARIABLES/OBJECTS ############

        self.cv_image = None                # colour image (from colorimagecb())
        self.depth_image = None             # aligned depth image (from depthimagecb())
        self.cam_info = None                # camera intrinsics (from caminfocb())
        self.depth_header = None            # header of the latest depth image

        # Confirmed ores, by transform name:
        # {name: {'kind': str, 'pos': [x, y, z], 'history': deque, 'hits': int, 'misses': int,
        #         'picked': bool, 'px': (u, v), 'depth': float}}
        self.tracked_ores = {}
        self.picked_ores = set()                                    # names reported picked on /ore_picked
        self.kind_counts = {kind: 0 for kind in ORE_TYPES}              # names handed out per kind so far
        self.pending = []                                           # new detections not yet named
        self.last_warn_time = 0.0
        self.last_debug_log_time = 0.0

        # Ore package (detected only once the eBot stands beside the arm)
        self.ebot_at_arm = False                                    # from ebotatarmcb()
        self.package_history = deque(maxlen=MEDIAN_FILTER_WINDOW)   # recent package positions
        self.package_pos = None                                     # smoothed position in base_link
        self.package_hits = 0
        self.package_picked = False                                 # set by ore_picked_cb()

        # Timers: the detection loop and the continuous transform broadcaster
        self.process_timer = self.create_timer(IMAGE_TIMER_PERIOD, self.process_image)
        self.broadcast_timer = self.create_timer(BROADCAST_TIMER_PERIOD, self.broadcast_transforms)

        self.get_logger().info('Task 3A Perception node initialized: ores on the conveyor, package on the eBot.')

    def colorimagecb(self, data):
        '''
        Description:    Callback function for the colour camera raw topic.
                        Converts the frame to an OpenCV BGR image.

        Args:
            data        (Image):    Input coloured raw image frame received from the image_raw camera topic

        Returns:
        '''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'colorimagecb error: {error}')

    def depthimagecb(self, data):
        '''
        Description:    Callback function for the aligned depth camera topic.
                        Stores the depth image and its header.

        Args:
            data        (Image):    Input depth image frame received from the aligned depth camera topic

        Returns:
        '''
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            self.depth_header = data.header
        except CvBridgeError as error:
            self.get_logger().error(f'depthimagecb error: {error}')

    def caminfocb(self, data):
        '''
        Description:    Callback function for the camera info topic. Stores the intrinsics.

        Args:
            data        (CameraInfo):   Camera intrinsics of the colour stream

        Returns:
        '''
        self.cam_info = data

    def ore_picked_cb(self, msg):
        '''
        Description:    Callback function for /ore_picked. The manipulation node reports an
                        ore it has picked, and its transform stops being published.

        Args:
            msg         (String):   Name of the picked ore, e.g. 'azurite_ore_1'

        Returns:
        '''
        ore_name = msg.data.strip()
        self.picked_ores.add(ore_name)
        if ore_name == PACKAGE_FRAME:
            self.package_picked = True
            self.get_logger().info('Ore package confirmed picked via topic. TF stopped.')
        if ore_name in self.tracked_ores:
            self.tracked_ores[ore_name]['picked'] = True
            self.get_logger().info(f'Ore {ore_name} confirmed picked via topic. TF stopped.')

    def ebotatarmcb(self, msg):
        '''
        Description:    Callback function for /ebot_at_arm. The planner reports that the eBot
                        stands beside the arm with the package, so package detection starts.

        Args:
            msg         (Bool):     True once the eBot has stopped beside the arm

        Returns:
        '''
        if msg.data and not self.ebot_at_arm:
            self.ebot_at_arm = True
            self.get_logger().info('eBot beside the arm: looking for the ore package')

    def detect_package(self, hsv):
        '''
        Description:    Finds the magenta patch on the package's top face, locates its centre in
                        base_link, and keeps a median-smoothed estimate of it.

        Args:
            hsv         (np.ndarray):   current colour frame in HSV

        Returns:
        '''
        mask = cv2.inRange(hsv, PACKAGE_COLOR_BOUNDS[0], PACKAGE_COLOR_BOUNDS[1])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # The largest blob that has the patch's shape and lies on the eBot's deck
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < PACKAGE_MIN_AREA or (best is not None and area <= best[0]):
                continue
            rect = cv2.minAreaRect(contour)
            (_, _), (box_width, box_height), _ = rect
            aspect_ratio = max(box_width, box_height) / max(min(box_width, box_height), 1e-5)
            fill_ratio = area / max(box_width * box_height, 1e-5)
            if aspect_ratio > PACKAGE_MAX_ASPECT_RATIO or fill_ratio < PACKAGE_MIN_FILL_RATIO:
                continue
            pixel_u, pixel_v = rect[0]
            depth = self._median_depth_eroded(cv2.boxPoints(rect), (pixel_u, pixel_v))
            point = self._deproject_to_base(pixel_u, pixel_v, depth, self.depth_header)
            if point is None:
                continue
            base_x, base_y, base_z = point
            if (PACKAGE_X_MIN <= base_x <= PACKAGE_X_MAX and PACKAGE_Y_MIN <= base_y <= PACKAGE_Y_MAX
                    and PACKAGE_Z_MIN <= base_z <= PACKAGE_Z_MAX):
                best = (area, np.array([base_x, base_y, base_z], dtype=np.float64), (pixel_u, pixel_v))

        if best is None:
            return
        self.package_history.append(best[1])
        self.package_pos = np.median(np.array(self.package_history), axis=0)
        self.package_hits += 1
        if self.package_hits == PACKAGE_CONFIRM_HITS:
            x, y, z = self.package_pos
            self.get_logger().info(
                f'[PERCEPTION] {PACKAGE_FRAME}: px=({best[2][0]:.1f}, {best[2][1]:.1f}), '
                f'base_link=({x:.3f}, {y:.3f}, {z:.3f})')

    def _median_depth_eroded(self, mask_poly, center_px):
        '''
        Description:    Median depth of the valid pixels on the ore's top face, eroded by one
                        pixel to leave out the edges, the black sides and the conveyor.

        Args:
            mask_poly   (np.ndarray):   corners of the ore's rectangle in pixels
            center_px   (tuple):        (u, v) centre of the rectangle, used as a fallback

        Returns:
            depth       (float):        median depth in metres, or None if no valid pixel
        '''
        if self.depth_image is None:
            return None

        height, width = self.depth_image.shape[:2]
        single_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(single_mask, [np.int32(mask_poly)], 255)

        # Erode the face once with a 3x3 kernel
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(single_mask, kernel, iterations=1)

        patch = self.depth_image.astype(np.float32)
        if self.depth_image.dtype == np.uint16:
            patch = patch / 1000.0          # millimetres -> metres

        valid_depths = patch[(eroded > 0) & np.isfinite(patch) & (patch > 0.2) & (patch < 2.5)]
        if valid_depths.size >= 4:
            return float(np.median(valid_depths))

        # Fallback: a small 7x7 patch around the rectangle's centre
        centre_col, centre_row = int(round(center_px[0])), int(round(center_px[1]))
        col_start, col_end = max(centre_col - 3, 0), min(centre_col + 4, width)
        row_start, row_end = max(centre_row - 3, 0), min(centre_row + 4, height)
        fallback_patch = patch[row_start:row_end, col_start:col_end]
        valid_fallback = fallback_patch[np.isfinite(fallback_patch) & (fallback_patch > 0.2) & (fallback_patch < 2.5)]
        if valid_fallback.size > 0:
            return float(np.median(valid_fallback))
        return None

    def _deproject_to_base(self, pixel_u, pixel_v, z, header):
        '''
        Description:    Back-projects a pixel and its depth into a 3D point in base_link.

        Args:
            pixel_u     (float):    pixel column
            pixel_v     (float):    pixel row
            z           (float):    depth at that pixel, in metres
            header      (Header):   header of the depth image (frame and stamp)

        Returns:
            point       (tuple):    (x, y, z) in base_link, in metres, or None if unavailable
        '''
        if self.cam_info is None or header is None or z is None:
            return None

        focal_x = float(self.cam_info.k[0])
        focal_y = float(self.cam_info.k[4])
        centre_x = float(self.cam_info.k[2])
        centre_y = float(self.cam_info.k[5])

        if focal_x <= 0.0 or focal_y <= 0.0:
            return None

        # pinhole model: pixel + depth -> point in the camera's optical frame
        pt_cam = PointStamped()
        pt_cam.header = header
        pt_cam.point.x = (float(pixel_u) - centre_x) * z / focal_x
        pt_cam.point.y = (float(pixel_v) - centre_y) * z / focal_y
        pt_cam.point.z = float(z)

        try:
            transform = self.tf_buffer.lookup_transform(BASE_FRAME, header.frame_id, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

        pt_base = do_transform_point(pt_cam, transform)
        return (float(pt_base.point.x), float(pt_base.point.y), float(pt_base.point.z))

    def process_image(self):
        '''
        Description:    Perception loop: detects coloured blobs, checks their shape, locates
                        them in base_link, and keeps a named track for each ore. Once the eBot
                        stands beside the arm, it also looks for the ore package.

        Args:

        Returns:
        '''
        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

        raw_candidates = self.find_ore_blobs(hsv, now_sec)
        roi_candidates = self.locate_candidates(raw_candidates)
        surviving_candidates = self.keep_best_four(roi_candidates, now_sec)
        matched_track_names, unmatched_candidates = self.match_tracks(surviving_candidates)
        self.promote_pending(unmatched_candidates)
        self.update_track_status(matched_track_names)
        self.log_tracks(now_sec)

        # The ore package, once the eBot stands beside the arm
        if self.ebot_at_arm and not self.package_picked:
            self.detect_package(hsv)

    def find_ore_blobs(self, hsv, now_sec):
        '''
        Description:    Colour threshold and shape check, per kind of ore, inside the conveyor
                        part of the image.

        Args:
            hsv             (np.ndarray):   current colour frame in HSV
            now_sec         (float):        current time in seconds

        Returns:
            raw_candidates  (list):         one dict per ore-shaped blob: kind, centre pixel,
                                            corner polygon, area, aspect ratio and fill ratio
        '''
        height, width = self.cv_image.shape[:2]

        roi_x0 = int(width * ROI_X_FRAC)
        roi_y0 = int(height * ROI_Y_MIN_FRAC)
        roi_y1 = int(height * ROI_Y_MAX_FRAC)

        raw_candidates = []
        for ore_type in ORE_TYPES:
            lower, upper = COLOR_BOUNDS[ore_type]
            mask = cv2.inRange(hsv, lower, upper)

            # Restrict to the conveyor part of the image
            mask[:roi_y0, :] = 0
            mask[roi_y1:, :] = 0
            mask[:, :roi_x0] = 0

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_AREA_BY_TYPE[ore_type]:
                    continue

                rect = cv2.minAreaRect(contour)
                (_, _), (box_width, box_height), _ = rect
                rect_area = max(box_width * box_height, 1e-5)
                aspect_ratio = max(box_width, box_height) / max(min(box_width, box_height), 1e-5)
                fill_ratio = area / rect_area
                if DEBUG_LOG and now_sec - getattr(self, '_last_area_log', 0.0) >= 1.0:
                    self._last_area_log = now_sec
                    self.get_logger().info(
                        f'[AREA] {ore_type}: area={area:.0f} ar={aspect_ratio:.2f} fill={fill_ratio:.2f}')
                if area <= MAX_CONTOUR_AREA:
                    # Single ore shape verification
                    if aspect_ratio <= MAX_ASPECT_RATIO and fill_ratio >= MIN_FILL_RATIO:
                        poly = cv2.boxPoints(rect)
                        raw_candidates.append({
                            'kind': ore_type,
                            'center_px': rect[0],
                            'poly': poly,
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'fill_ratio': fill_ratio,
                        })
        return raw_candidates

    def locate_candidates(self, raw_candidates):
        '''
        Description:    Locates each candidate in base_link and keeps only those on the conveyor.

        Args:
            raw_candidates  (list):     candidates from find_ore_blobs()

        Returns:
            roi_candidates  (list):     the candidates on the conveyor, each with 'pos_base'
                                        (x, y, z in base_link) and 'depth' added
        '''
        roi_candidates = []
        for candidate in raw_candidates:
            pixel_u, pixel_v = candidate['center_px']
            depth = self._median_depth_eroded(candidate['poly'], (pixel_u, pixel_v))
            if depth is None:
                continue

            pt_base = self._deproject_to_base(pixel_u, pixel_v, depth, self.depth_header)
            if pt_base is None:
                continue

            base_x, base_y, base_z = pt_base

            # Fixed scene geometry: the ore must lie inside the conveyor box
            if not (CONVEYOR_BELT_X_MIN <= base_x <= CONVEYOR_BELT_X_MAX):
                continue
            if not (CONVEYOR_BELT_Y_MIN <= base_y <= CONVEYOR_BELT_Y_MAX):
                continue
            if not (CONVEYOR_BELT_Z_MIN <= base_z <= CONVEYOR_BELT_Z_MAX):
                continue

            candidate['pos_base'] = np.array([base_x, base_y, base_z], dtype=np.float64)
            candidate['depth'] = depth
            roi_candidates.append(candidate)
        return roi_candidates

    def keep_best_four(self, roi_candidates, now_sec):
        '''
        Description:    Four ores are spawned: if more candidates survive, keeps the four most
                        ore-like (most square and solid).

        Args:
            roi_candidates  (list):     candidates from locate_candidates()
            now_sec         (float):    current time in seconds

        Returns:
            surviving_candidates    (list):     at most four candidates
        '''
        if len(roi_candidates) > 4:
            def candidate_quality(candidate):
                '''
                Description:    Score of a candidate: lower is more square and solid.

                Args:
                    candidate   (dict):     candidate with 'aspect_ratio' and 'fill_ratio'

                Returns:
                    score       (float):    weighted distance from a square, well-filled blob
                '''
                ar_err = abs(candidate['aspect_ratio'] - 1.0)
                fill_err = abs(candidate['fill_ratio'] - 0.85)
                return 2.0 * ar_err + fill_err

            roi_candidates.sort(key=candidate_quality)
            return roi_candidates[:4]

        if len(roi_candidates) < 4 and len(self.tracked_ores) < 4:   # picked ores leave the view
            if now_sec - self.last_warn_time >= 1.0:
                self.get_logger().warn(
                    f'Detected {len(roi_candidates)}/4 ores on conveyor; maintaining tracked frames.'
                )
                self.last_warn_time = now_sec
        return roi_candidates

    def match_tracks(self, surviving_candidates):
        '''
        Description:    Matches candidates to existing tracks by nearest neighbour in base_link,
                        and median-smooths each matched track's position.

        Args:
            surviving_candidates    (list):     candidates from keep_best_four()

        Returns:
            matches         (tuple):    (set of matched track names, list of unmatched candidates)
        '''
        matched_track_names = set()
        unmatched_candidates = []

        for candidate in surviving_candidates:
            best_track_name = None
            best_dist = 1e9

            for name, track in self.tracked_ores.items():
                if track['picked'] or track['kind'] != candidate['kind']:
                    continue
                if name in matched_track_names:
                    continue

                distance = float(np.linalg.norm(candidate['pos_base'] - track['pos']))
                if distance <= TRACK_GATE_M and distance < best_dist:
                    best_dist = distance
                    best_track_name = name

            if best_track_name is not None:
                matched_track_names.add(best_track_name)
                track = self.tracked_ores[best_track_name]
                track['history'].append(candidate['pos_base'])
                # Median-smooth the position over the last few frames
                track['pos'] = np.median(np.array(track['history']), axis=0)
                track['hits'] += 1
                track['misses'] = 0
                track['px'] = candidate['center_px']
                track['depth'] = candidate['depth']
            else:
                unmatched_candidates.append(candidate)
        return matched_track_names, unmatched_candidates

    def promote_pending(self, unmatched_candidates):
        '''
        Description:    A new ore must be seen PROMOTE_HITS times in a row before it gets a
                        name. Updates the pending tracks with the unmatched candidates, then
                        names the confirmed ones in a fixed order, up to four ores in all.

        Args:
            unmatched_candidates    (list):     candidates no existing track matched

        Returns:
        '''
        # Sort deterministically by base_link y, then x
        unmatched_candidates.sort(key=lambda candidate: (candidate['pos_base'][1], candidate['pos_base'][0]))

        for pending_track in self.pending:
            pending_track['seen'] = False
        for candidate in unmatched_candidates:
            best_pending, best_pending_dist = None, 1e9
            for pending_track in self.pending:
                if pending_track['kind'] != candidate['kind'] or pending_track['seen']:
                    continue
                distance = float(np.linalg.norm(candidate['pos_base'] - pending_track['pos']))
                if distance <= TRACK_GATE_M and distance < best_pending_dist:
                    best_pending, best_pending_dist = pending_track, distance
            if best_pending is not None:
                best_pending.update(seen=True, pos=candidate['pos_base'], candidate=candidate)
                best_pending['hits'] += 1
            else:
                self.pending.append({'kind': candidate['kind'], 'pos': candidate['pos_base'],
                                     'candidate': candidate, 'hits': 1, 'seen': True})
        self.pending = [pending_track for pending_track in self.pending if pending_track['seen']]

        # Name the confirmed ones, in a fixed order, up to four in all
        ready = sorted([pending_track for pending_track in self.pending if pending_track['hits'] >= PROMOTE_HITS],
                       key=lambda pending_track: (pending_track['pos'][1], pending_track['pos'][0]))
        for pending_track in ready:
            if len(self.tracked_ores) >= 4:
                self.pending.clear()
                break
            self.pending.remove(pending_track)
            kind, candidate = pending_track['kind'], pending_track['candidate']
            self.kind_counts[kind] += 1
            new_name = f'{kind}_{self.kind_counts[kind]}'
            hist = deque(maxlen=MEDIAN_FILTER_WINDOW)
            hist.append(candidate['pos_base'])
            self.tracked_ores[new_name] = {
                'kind': kind, 'pos': candidate['pos_base'].copy(), 'history': hist,
                'hits': CONFIRM_HITS, 'misses': 0, 'picked': False,
                'px': candidate['center_px'], 'depth': candidate['depth'],
            }

    def update_track_status(self, matched_track_names):
        '''
        Description:    Counts missed detections, and stops the tracks whose ore has lifted off
                        the conveyor.

        Args:
            matched_track_names     (set):      names of the tracks seen in this frame

        Returns:
        '''
        for name, track in self.tracked_ores.items():
            if track['picked']:
                continue
            if name not in matched_track_names:
                track['misses'] += 1
            if track['pos'][2] > (CONVEYOR_BELT_Z_MAX + LIFT_OFF_Z_THRESHOLD):
                track['picked'] = True
                self.get_logger().info(f'{name} lifted off conveyor (z={track["pos"][2]:.3f}m). TF stopped.')

    def log_tracks(self, now_sec):
        '''
        Description:    Debug logging of every confirmed track, once per second.

        Args:
            now_sec         (float):    current time in seconds

        Returns:
        '''
        if DEBUG_LOG and (now_sec - self.last_debug_log_time >= 1.0):
            for name, track in self.tracked_ores.items():
                if not track['picked'] and track['hits'] >= CONFIRM_HITS:
                    pixel_x, pixel_y = track['px']
                    x, y, z = track['pos']
                    self.get_logger().info(
                        f'[PERCEPTION] {name}: px=({pixel_x:.1f}, {pixel_y:.1f}), depth={track["depth"]:.3f}m, '
                        f'base_link=({x:.3f}, {y:.3f}, {z:.3f})'
                    )
            self.last_debug_log_time = now_sec

    def broadcast_transforms(self):
        '''
        Description:    Broadcasts a transform for every confirmed ore not yet picked, and for
                        the ore package while it is on the eBot beside the arm.

        Args:

        Returns:
        '''
        now = self.get_clock().now().to_msg()
        for name, data in self.tracked_ores.items():
            if data['picked'] or name in self.picked_ores:
                continue
            if data['hits'] < CONFIRM_HITS:
                continue

            ore_tf = TransformStamped()
            ore_tf.header.stamp = now
            ore_tf.header.frame_id = BASE_FRAME
            ore_tf.child_frame_id = name
            ore_tf.transform.translation.x = float(data['pos'][0])
            ore_tf.transform.translation.y = float(data['pos'][1])
            ore_tf.transform.translation.z = float(data['pos'][2])
            # Identity rotation: only the position is used
            ore_tf.transform.rotation.x = 0.0
            ore_tf.transform.rotation.y = 0.0
            ore_tf.transform.rotation.z = 0.0
            ore_tf.transform.rotation.w = 1.0

            self.broadcaster.sendTransform(ore_tf)

        if self.package_pos is not None and not self.package_picked and self.package_hits >= PACKAGE_CONFIRM_HITS:
            package_tf = TransformStamped()
            package_tf.header.stamp = now
            package_tf.header.frame_id = BASE_FRAME
            package_tf.child_frame_id = PACKAGE_FRAME
            package_tf.transform.translation.x = float(self.package_pos[0])
            package_tf.transform.translation.y = float(self.package_pos[1])
            package_tf.transform.translation.z = float(self.package_pos[2])
            package_tf.transform.rotation.w = 1.0
            self.broadcaster.sendTransform(package_tf)


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ore_perception class to perform its task
    '''

    rclpy.init()                       # initialisation

    node = ore_perception()                     # creating a new object for class 'ore_perception'

    try:
        rclpy.spin(node)                        # spinning on the object to make it alive in ROS 2 DDS
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()                     # destroy node after spin ends
        if rclpy.ok():
            rclpy.shutdown()                    # shutdown process


if __name__ == '__main__':

    main()
