#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 2A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task2A_perception.py
# Functions:
#                   main
# Nodes:            Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw,
#                                          /camera/camera/aligned_depth_to_color/image_raw,
#                                          /camera/camera/color/camera_info, /ore_picked ]


################### IMPORT MODULES #######################

from collections import deque
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
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
                    base_link -> <kind>_<n> transform until the ore is picked.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ore_perception
        '''

        super().__init__(
            'task2A_perception_node',
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
        self.kind_counts = {ot: 0 for ot in ORE_TYPES}              # names handed out per kind so far
        self.pending = []                                           # new detections not yet named
        self.last_warn_time = 0.0
        self.last_debug_log_time = 0.0

        # Timers: the detection loop and the continuous transform broadcaster
        self.process_timer = self.create_timer(IMAGE_TIMER_PERIOD, self.process_image)
        self.broadcast_timer = self.create_timer(BROADCAST_TIMER_PERIOD, self.broadcast_transforms)

        self.get_logger().info('Task 2A Perception node initialized with geometric ROI & square shape validation.')

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
        except CvBridgeError as e:
            self.get_logger().error(f'colorimagecb error: {e}')

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
        except CvBridgeError as e:
            self.get_logger().error(f'depthimagecb error: {e}')

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
        if ore_name in self.tracked_ores:
            self.tracked_ores[ore_name]['picked'] = True
            self.get_logger().info(f'Ore {ore_name} confirmed picked via topic. TF stopped.')

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

        h, w = self.depth_image.shape[:2]
        single_mask = np.zeros((h, w), dtype=np.uint8)
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
        cx, cy = int(round(center_px[0])), int(round(center_px[1]))
        x0, x1 = max(cx - 3, 0), min(cx + 4, w)
        y0, y1 = max(cy - 3, 0), min(cy + 4, h)
        fallback_patch = patch[y0:y1, x0:x1]
        valid_fallback = fallback_patch[np.isfinite(fallback_patch) & (fallback_patch > 0.2) & (fallback_patch < 2.5)]
        if valid_fallback.size > 0:
            return float(np.median(valid_fallback))
        return None

    def _deproject_to_base(self, u, v, z, header):
        '''
        Description:    Back-projects a pixel and its depth into a 3D point in base_link.

        Args:
            u           (float):    pixel column
            v           (float):    pixel row
            z           (float):    depth at that pixel, in metres
            header      (Header):   header of the depth image (frame and stamp)

        Returns:
            point       (tuple):    (x, y, z) in base_link, in metres, or None if unavailable
        '''
        if self.cam_info is None or header is None or z is None:
            return None

        fx = float(self.cam_info.k[0])
        fy = float(self.cam_info.k[4])
        cx = float(self.cam_info.k[2])
        cy = float(self.cam_info.k[5])

        if fx <= 0.0 or fy <= 0.0:
            return None

        # pinhole model: pixel + depth -> point in the camera's optical frame
        pt_cam = PointStamped()
        pt_cam.header = header
        pt_cam.point.x = (float(u) - cx) * z / fx
        pt_cam.point.y = (float(v) - cy) * z / fy
        pt_cam.point.z = float(z)

        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, header.frame_id, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

        pt_base = do_transform_point(pt_cam, tf)
        return (float(pt_base.point.x), float(pt_base.point.y), float(pt_base.point.z))

    def process_image(self):
        '''
        Description:    Perception loop: detects coloured blobs, checks their shape, locates
                        them in base_link, and keeps a named track for each ore.

        Args:

        Returns:
        '''
        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        height, width = self.cv_image.shape[:2]

        roi_x0 = int(width * ROI_X_FRAC)
        roi_y0 = int(height * ROI_Y_MIN_FRAC)
        roi_y1 = int(height * ROI_Y_MAX_FRAC)

        raw_candidates = []

        # ------------------------------------------------------------------
        # Colour threshold and shape check, per kind of ore
        # ------------------------------------------------------------------
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

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_AREA_BY_TYPE[ore_type]:
                    continue

                rect = cv2.minAreaRect(cnt)
                (_, _), (w, h), _ = rect
                rect_area = max(w * h, 1e-5)
                aspect_ratio = max(w, h) / max(min(w, h), 1e-5)
                fill_ratio = area / rect_area
                if DEBUG_LOG and now_sec - getattr(self, '_last_area_log', 0.0) >= 1.0:
                    self._last_area_log = now_sec
                    self.get_logger().info(f'[AREA] {ore_type}: area={area:.0f} ar={aspect_ratio:.2f} fill={fill_ratio:.2f}')
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

        # ------------------------------------------------------------------
        # Locate each candidate in base_link and keep only those on the conveyor
        # ------------------------------------------------------------------
        roi_candidates = []
        for cand in raw_candidates:
            u, v = cand['center_px']
            depth = self._median_depth_eroded(cand['poly'], (u, v))
            if depth is None:
                continue

            pt_base = self._deproject_to_base(u, v, depth, self.depth_header)
            if pt_base is None:
                continue

            bx, by, bz = pt_base

            # Fixed scene geometry: the ore must lie inside the conveyor box
            if not (CONVEYOR_BELT_X_MIN <= bx <= CONVEYOR_BELT_X_MAX):
                continue
            if not (CONVEYOR_BELT_Y_MIN <= by <= CONVEYOR_BELT_Y_MAX):
                continue
            if not (CONVEYOR_BELT_Z_MIN <= bz <= CONVEYOR_BELT_Z_MAX):
                continue

            cand['pos_base'] = np.array([bx, by, bz], dtype=np.float64)
            cand['depth'] = depth
            roi_candidates.append(cand)

        # ------------------------------------------------------------------
        # Four ores are spawned: if more candidates survive, keep the four most ore-like
        # ------------------------------------------------------------------
        if len(roi_candidates) > 4:
            def candidate_quality(c):
                '''
                Description:    Score of a candidate: lower is more square and solid.

                Args:
                    c           (dict):     candidate with 'aspect_ratio' and 'fill_ratio'

                Returns:
                    score       (float):    weighted distance from a square, well-filled blob
                '''
                ar_err = abs(c['aspect_ratio'] - 1.0)
                fill_err = abs(c['fill_ratio'] - 0.85)
                return 2.0 * ar_err + fill_err

            roi_candidates.sort(key=candidate_quality)
            surviving_candidates = roi_candidates[:4]
        else:
            surviving_candidates = roi_candidates
            if len(surviving_candidates) < 4:
                if now_sec - self.last_warn_time >= 1.0:
                    self.get_logger().warn(
                        f'Detected {len(surviving_candidates)}/4 ores on conveyor; maintaining tracked frames.'
                    )
                    self.last_warn_time = now_sec

        # ------------------------------------------------------------------
        # Match candidates to existing tracks by nearest neighbour in base_link
        # ------------------------------------------------------------------
        matched_track_names = set()
        unmatched_candidates = []

        for cand in surviving_candidates:
            best_track_name = None
            best_dist = 1e9

            for name, track in self.tracked_ores.items():
                if track['picked'] or track['kind'] != cand['kind']:
                    continue
                if name in matched_track_names:
                    continue

                distance = float(np.linalg.norm(cand['pos_base'] - track['pos']))
                if distance <= TRACK_GATE_M and distance < best_dist:
                    best_dist = distance
                    best_track_name = name

            if best_track_name is not None:
                matched_track_names.add(best_track_name)
                track = self.tracked_ores[best_track_name]
                track['history'].append(cand['pos_base'])
                # Median-smooth the position over the last few frames
                track['pos'] = np.median(np.array(track['history']), axis=0)
                track['hits'] += 1
                track['misses'] = 0
                track['px'] = cand['center_px']
                track['depth'] = cand['depth']
            else:
                unmatched_candidates.append(cand)

        # Unmatched candidates: sort deterministically by base_link y, then x
        unmatched_candidates.sort(key=lambda c: (c['pos_base'][1], c['pos_base'][0]))

        # A new ore must be seen PROMOTE_HITS times in a row before it gets a name
        for p in self.pending:
            p['seen'] = False
        for cand in unmatched_candidates:
            best_pending, best_pending_dist = None, 1e9
            for p in self.pending:
                if p['kind'] != cand['kind'] or p['seen']:
                    continue
                distance = float(np.linalg.norm(cand['pos_base'] - p['pos']))
                if distance <= TRACK_GATE_M and distance < best_pending_dist:
                    best_pending, best_pending_dist = p, distance
            if best_pending is not None:
                best_pending.update(seen=True, pos=cand['pos_base'], cand=cand)
                best_pending['hits'] += 1
            else:
                self.pending.append({'kind': cand['kind'], 'pos': cand['pos_base'], 'cand': cand, 'hits': 1, 'seen': True})
        self.pending = [p for p in self.pending if p['seen']]

        # Name the confirmed ones, in a fixed order, up to four in all
        ready = sorted([p for p in self.pending if p['hits'] >= PROMOTE_HITS], key=lambda p: (p['pos'][1], p['pos'][0]))
        for p in ready:
            if len(self.tracked_ores) >= 4:
                self.pending.clear()
                break
            self.pending.remove(p)
            kind, cand = p['kind'], p['cand']
            self.kind_counts[kind] += 1
            new_name = f'{kind}_{self.kind_counts[kind]}'
            hist = deque(maxlen=MEDIAN_FILTER_WINDOW)
            hist.append(cand['pos_base'])
            self.tracked_ores[new_name] = {
                'kind': kind, 'pos': cand['pos_base'].copy(), 'history': hist,
                'hits': CONFIRM_HITS, 'misses': 0, 'picked': False,
                'px': cand['center_px'], 'depth': cand['depth'],
            }

        # Count missed detections, and stop tracks whose ore has lifted off the conveyor
        for name, track in self.tracked_ores.items():
            if track['picked']:
                continue
            if name not in matched_track_names:
                track['misses'] += 1
            if track['pos'][2] > (CONVEYOR_BELT_Z_MAX + LIFT_OFF_Z_THRESHOLD):
                track['picked'] = True
                self.get_logger().info(f'{name} lifted off conveyor (z={track["pos"][2]:.3f}m). TF stopped.')

        # ------------------------------------------------------------------
        # Debug logging, once per second
        # ------------------------------------------------------------------
        if DEBUG_LOG and (now_sec - self.last_debug_log_time >= 1.0):
            for name, track in self.tracked_ores.items():
                if not track['picked'] and track['hits'] >= CONFIRM_HITS:
                    cx, cy = track['px']
                    x, y, z = track['pos']
                    self.get_logger().info(
                        f'[PERCEPTION] {name}: px=({cx:.1f}, {cy:.1f}), depth={track["depth"]:.3f}m, '
                        f'base_link=({x:.3f}, {y:.3f}, {z:.3f})'
                    )
            self.last_debug_log_time = now_sec

    def broadcast_transforms(self):
        '''
        Description:    Broadcasts a transform for every confirmed ore not yet picked.

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
