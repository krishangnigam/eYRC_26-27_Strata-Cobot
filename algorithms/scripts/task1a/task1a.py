#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task1a.py
# Functions:        detect_ores, rect_overlap, rect_distance, place_label, ore_tf.__init__, ore_tf.colorimagecb,
#                   ore_tf.depthimagecb, ore_tf.caminfocb, ore_tf.median_depth,
#                   ore_tf.deproject_to_base, ore_tf.match_tracks, ore_tf.plausible,
#                   ore_tf.locate_detections, ore_tf.update_tracks, ore_tf.publish_tracks,
#                   ore_tf.draw_and_save, ore_tf.process_image, main
# Nodes:            ore_tf_publisher
#
# Publishing Topics  - [ /tf ]
# Subscribing Topics - [ /camera/camera/color/image_raw,
#                        /camera/camera/aligned_depth_to_color/image_raw,
#                        /camera/camera/color/camera_info ]


################### IMPORT MODULES #######################

import math
import sys

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point


##################### TASK CONSTANTS #######################

team_id = 6494

# Two ores of each type are spawned - six in all - told apart by an id of 1 or 2.
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']
ids_per_type = 2

color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

base_frame = 'base_link'
window_name = 'Task 1A - Ore Detection'
detection_image = f'SC#{team_id}_task1A_detection.png'


##################### DETECTION TUNING #######################

# HSV bounds (H 0-179, S/V 0-255) measured from the simulation.
color_bounds = {
    'azurite_ore':    (np.array([100, 180, 150]), np.array([112, 255, 255])),   # blue
    'malachite_ore':  (np.array([50, 140, 50]), np.array([90, 255, 255])),      # green
    'vanadinite_ore': (np.array([5, 120, 120]), np.array([16, 255, 255])),      # orange
}

# Smallest coloured blob kept, per type: a face can be partly hidden by another ore.
min_contour_area = {'azurite_ore': 20, 'malachite_ore': 20, 'vanadinite_ore': 15}

# Conveyor region of the image, as fractions of width/height (scene-specific).
roi_x_frac = 0.50
roi_y_frac = (0.40, 0.99)

# Height band of the conveyor's ore plane in base_link (observed z ~ -0.065 m).
# Rejects orange false positives seen elsewhere in the cell (z ~ -0.55 m).
conveyor_z_range = (-0.15, 0.00)

# Tracking: keeps azurite_ore_1 vs azurite_ore_2 (etc.) stable for the whole run.
match_dist_px = 15              # matching radius for an unconfirmed candidate
confirmed_match_dist_px = 8     # tighter radius once a track is confirmed
confirm_hits = 3                # net frames a candidate must survive to be confirmed
provisional_miss_limit = 2      # misses before an unconfirmed candidate is dropped
cluster_radius_m = 0.40         # a new ore must lie this close to an already-confirmed one

image_processing_rate = 0.2     # seconds between processing cycles
depth_window_px = 4             # half-width of the window the depth median is taken over

marker_radius_px = 14            # radius of the circle drawn round each ore

# Debug helper: click the detection window to log a pixel's BGR/HSV.
debug_hsv_click = False


##################### FUNCTION DEFINITIONS #######################

def detect_ores(image):
    '''
    Description:    Function to detect the ores present in a colour image frame and
                    return the pixel location and the type of each one found.

    Args:
        image           (np.ndarray):   BGR colour frame from the camera topic

    Returns:
        center_ore_list (list):         Centre pixel (cX, cY) of every ore detected
        ore_type_list   (list):         Type of each ore detected, from 'ore_types'
    '''
    center_ore_list, ore_type_list = [], []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    height, width = image.shape[:2]
    x0 = int(width * roi_x_frac)
    y0, y1 = int(height * roi_y_frac[0]), int(height * roi_y_frac[1])

    for ore_type in ore_types:
        lower, upper = color_bounds[ore_type]
        mask = cv2.inRange(hsv, lower, upper)
        mask[:y0, :] = 0                                    # keep only the conveyor region
        mask[y1:, :] = 0
        mask[:, :x0] = 0
        # mild cleanup only: an aggressive opening erases a partly hidden face
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area[ore_type]:
                continue
            m = cv2.moments(contour)
            if m['m00'] == 0:
                continue
            center_ore_list.append((int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])))
            ore_type_list.append(ore_type)

    return center_ore_list, ore_type_list


def rect_overlap(a, b):
    '''
    Description:    Overlapping area of two rectangles (x0, y0, x1, y1), in pixels.
    '''
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def rect_distance(rect, point):
    '''
    Description:    Distance from a point to the nearest edge of a rectangle (0 if inside).
    '''
    x = min(max(point[0], rect[0]), rect[2])
    y = min(max(point[1], rect[1]), rect[3])
    return math.dist((x, y), point)


def place_label(text, anchor, frame_shape, taken, ores, font_scale=0.45, thickness=1):
    '''
    Description:    Picks a spot for a label beside its ore: inside the frame, clear of
                    other labels and of every ore marker, and nearer its own ore than any
                    other, so a reader cannot pair it with the wrong ore.

    Args:
        text        (str):      Label text
        anchor      (tuple):    (x, y) pixel of this label's ore
        frame_shape (tuple):    Shape of the frame being drawn on
        taken       (list):     Rectangles of labels already placed (appended to)
        ores        (list):     (x, y) pixels of every ore being labelled

    Returns:
        tuple:                  (x, y) baseline-left position for cv2.putText
    '''
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    h, w = frame_shape[:2]
    ax, ay = anchor
    pad = marker_radius_px + 4
    markers = [(x - pad, y - pad, x + pad, y + pad) for x, y in ores]
    offsets = [(dx, dy) for ring in range(0, 5)
               for dx, dy in ((pad, -6 - ring * (th + 6)), (pad, th + 6 + ring * (th + 6)),
                              (-tw - pad, -6 - ring * (th + 6)), (-tw - pad, th + 6 + ring * (th + 6)),
                              (-tw // 2, -pad - ring * (th + 6)), (-tw // 2, pad + th + ring * (th + 6)))]
    best = None
    for rank, (dx, dy) in enumerate(offsets):
        x = min(max(ax + dx, 2), w - tw - 2)                # clamp inside the frame
        y = min(max(ay + dy, th + 2), h - base - 2)
        rect = (x - 2, y - th - 2, x + tw + 2, y + base + 2)
        own = rect_distance(rect, anchor)
        wrong_ore = any(rect_distance(rect, o) < own for o in ores if o != tuple(anchor))
        cost = (sum(rect_overlap(rect, r) for r in taken) * 10 +
                sum(rect_overlap(rect, m) for m in markers) * 10 +
                (100000 if wrong_ore else 0) + own)         # then simply: as close as possible
        if best is None or (cost, rank) < best[:2]:
            best = (cost, rank, x, y, rect)
    taken.append(best[4])
    return best[2], best[3]


##################### CLASS DEFINITION #######################

class ore_tf(Node):
    '''
    ___CLASS___

    Description:    Detects the ores on the conveyor, tracks each one under a stable name,
                    and broadcasts its position as a tf2 transform in base_link.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ore_tf
        '''
        super().__init__(
            'ore_tf_publisher',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic SUBSCRIPTIONS ############
        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)

        ############ TF ############
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)

        ############ State Variables ############
        self.cv_image = None
        self.depth_image = None
        self.cam_info = None
        self.color_header = None

        # per type: list of tracks {id, px, hits, misses, confirmed, point}
        self.ore_tracks = {ore_type: [] for ore_type in ore_types}
        self.confirmed_points = []                          # 3D points of confirmed ores
        self.image_saved = False

        ############ Display ############
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        if debug_hsv_click:
            cv2.setMouseCallback(window_name, self.on_mouse)

        ############ Control Timer ############
        self.timer = self.create_timer(image_processing_rate, self.process_image)
        self.get_logger().info('Ore detector running on simulation clock.')

    # ------------------------------------------------------------------ callbacks

    def colorimagecb(self, data):
        '''
        Description:    Stores the latest colour frame and its header.

        Args:
            data (Image):   Colour frame from the camera
        '''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            self.color_header = data.header
        except CvBridgeError as e:
            self.get_logger().error(f'colorimagecb: {e}')

    def depthimagecb(self, data):
        '''
        Description:    Stores the latest depth frame, aligned to the colour frame.

        Args:
            data (Image):   Aligned depth frame from the camera
        '''
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'depthimagecb: {e}')

    def caminfocb(self, data):
        '''
        Description:    Stores the camera intrinsics (never hard-coded).

        Args:
            data (CameraInfo):  Camera calibration
        '''
        self.cam_info = data

    def on_mouse(self, event, x, y, flags, param):
        '''
        Description:    Debug helper: logs the BGR/HSV of a clicked pixel.
        '''
        if event != cv2.EVENT_LBUTTONDOWN or self.cv_image is None:
            return
        if 0 <= y < self.cv_image.shape[0] and 0 <= x < self.cv_image.shape[1]:
            b, g, r = self.cv_image[y, x]
            h, s, v = cv2.cvtColor(self.cv_image[y:y + 1, x:x + 1], cv2.COLOR_BGR2HSV)[0, 0]
            self.get_logger().info(f'({x},{y}) BGR=({b},{g},{r}) HSV=({h},{s},{v})')

    # ------------------------------------------------------------------ geometry

    def median_depth(self, cx, cy):
        '''
        Description:    Median of the valid depth readings around a pixel.

        Args:
            cx, cy (int):   Pixel coordinates

        Returns:
            float or None:  Depth in metres, or None when no reading is valid
        '''
        h, w = self.depth_image.shape[:2]
        k = depth_window_px
        patch = self.depth_image[max(cy - k, 0):min(cy + k + 1, h),
                                 max(cx - k, 0):min(cx + k + 1, w)].astype(np.float32)
        if self.depth_image.dtype == np.uint16:
            patch /= 1000.0                                 # 16UC1 is millimetres
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        return float(np.median(valid)) if valid.size else None

    def deproject_to_base(self, u, v, z):
        '''
        Description:    Back-projects pixel (u, v) at depth z into base_link.

        Args:
            u, v (int):     Pixel coordinates
            z (float):      Depth in metres

        Returns:
            tuple or None:  (x, y, z) in base_link, or None when TF is not ready
        '''
        fx, fy = self.cam_info.k[0], self.cam_info.k[4]
        cx, cy = self.cam_info.k[2], self.cam_info.k[5]
        if fx <= 0.0 or fy <= 0.0:
            return None
        point = PointStamped()
        point.header = self.color_header
        point.point.x = (u - cx) * z / fx
        point.point.y = (v - cy) * z / fy
        point.point.z = float(z)
        try:
            tf = self.tf_buffer.lookup_transform(base_frame, self.color_header.frame_id, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        p = do_transform_point(point, tf).point
        return (float(p.x), float(p.y), float(p.z))

    # ------------------------------------------------------------------ tracking

    def match_tracks(self, ore_type, detections_px):
        '''
        Description:    One-to-one, nearest-first matching of this frame's detections to
                        the type's tracks; ages unmatched candidates and opens new ones
                        for free ids.

        Args:
            ore_type      (str):    One of 'ore_types'
            detections_px (list):   (cX, cY) of each detection of that type

        Returns:
            dict:                   detection index -> matched track
        '''
        tracks = self.ore_tracks[ore_type]
        pairs = sorted(
            (math.dist(tr['px'], px), ti, di)
            for ti, tr in enumerate(tracks) for di, px in enumerate(detections_px)
            if math.dist(tr['px'], px) <= (confirmed_match_dist_px if tr['confirmed'] else match_dist_px))

        used_tracks, used_dets, matched = set(), set(), {}
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            tr = tracks[ti]
            tr['px'] = detections_px[di]
            tr['hits'] += 1
            tr['misses'] = 0
            matched[di] = tr

        # unconfirmed candidates that missed decay; stale ones are dropped
        for ti, tr in enumerate(tracks):
            if ti not in used_tracks and not tr['confirmed']:
                tr['misses'] += 1
                tr['hits'] = max(0, tr['hits'] - 1)
        tracks[:] = [tr for tr in tracks if tr['confirmed'] or tr['misses'] <= provisional_miss_limit]

        # unmatched detections open a new candidate under a free id
        used_ids = {tr['id'] for tr in tracks}
        for di, px in enumerate(detections_px):
            if di in used_dets or len(tracks) >= ids_per_type:
                continue
            free_id = next(i for i in range(1, ids_per_type + 1) if i not in used_ids)
            tr = {'id': free_id, 'px': px, 'hits': 1, 'misses': 0, 'confirmed': False, 'point': None}
            tracks.append(tr)
            used_ids.add(free_id)
            matched[di] = tr
        return matched

    def plausible(self, point):
        '''
        Description:    True when 'point' lies near an already-confirmed ore, or when none
                        is confirmed yet. Rejects static colour false positives.

        Args:
            point (tuple):  (x, y, z) in base_link
        '''
        return not self.confirmed_points or any(
            math.dist(point, p) <= cluster_radius_m for p in self.confirmed_points)

    # ------------------------------------------------------------------ pipeline

    def locate_detections(self, frame):
        '''
        Description:    Detects the ores and back-projects each to base_link, keeping only
                        detections on the conveyor's height band.

        Args:
            frame (np.ndarray):     Colour frame

        Returns:
            dict:                   ore_type -> list of {'px', 'point'}
        '''
        located = {ore_type: [] for ore_type in ore_types}
        for (cx, cy), ore_type in zip(*detect_ores(frame)):
            z = self.median_depth(cx, cy)
            point = self.deproject_to_base(cx, cy, z) if z is not None else None
            if point is not None and conveyor_z_range[0] <= point[2] <= conveyor_z_range[1]:
                located[ore_type].append({'px': (cx, cy), 'point': point})
        return located

    def update_tracks(self, located):
        '''
        Description:    Matches detections to tracks, refreshes their 3D points, and
                        confirms candidates that have earned it.

        Args:
            located (dict):     Output of locate_detections()
        '''
        for ore_type, detections in located.items():
            matched = self.match_tracks(ore_type, [d['px'] for d in detections])
            for di, tr in matched.items():
                tr['point'] = detections[di]['point']
                if tr['confirmed'] or tr['hits'] < confirm_hits:
                    continue
                if self.plausible(tr['point']):
                    tr['confirmed'] = True
                    self.confirmed_points.append(tr['point'])
                    self.get_logger().info(f'{ore_type}_{tr["id"]} confirmed at '
                                           f'({tr["point"][0]:.3f}, {tr["point"][1]:.3f}, {tr["point"][2]:.3f})')
                else:
                    tr['hits'] = 0                          # must earn confirmation again

    def confirmed_tracks(self):
        '''
        Description:    Every confirmed track with a known position, with its frame name.

        Returns:
            list:   (name, track) pairs
        '''
        return [(f'{ore_type}_{tr["id"]}', tr) for ore_type in ore_types
                for tr in self.ore_tracks[ore_type] if tr['confirmed'] and tr['point'] is not None]

    def publish_tracks(self):
        '''
        Description:    Broadcasts every confirmed ore on every cycle, from its last fix.
        '''
        stamp = self.get_clock().now().to_msg()
        for name, tr in self.confirmed_tracks():
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = base_frame
            t.child_frame_id = name
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = tr['point']
            t.transform.rotation.w = 1.0                    # identity: a legal quaternion
            self.br.sendTransform(t)

    def draw_and_save(self, frame):
        '''
        Description:    Draws a boundary and a readable label on every confirmed ore,
                        shows the frame, and saves it at full camera resolution once all
                        six ores are confirmed (or when 's' is pressed).

        Args:
            frame (np.ndarray):     Colour frame to annotate (modified in place)
        '''
        taken = []
        tracks = self.confirmed_tracks()
        for name, tr in tracks:
            cx, cy = tr['px']
            cv2.circle(frame, (cx, cy), marker_radius_px, (0, 255, 0), 2)
        ores = [tuple(tr['px']) for _, tr in tracks]
        for name, tr in tracks:
            x, y = place_label(name, tr['px'], frame.shape, taken, ores)
            x0, y0, x1, y1 = taken[-1]
            near = (min(max(tr['px'][0], x0), x1), min(max(tr['px'][1], y0), y1))
            if math.dist(near, tr['px']) > 22:                  # label pushed away: connect it
                cv2.line(frame, tr['px'], near, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        all_found = len(tracks) == len(ore_types) * ids_per_type
        if key == ord('s') or (all_found and not self.image_saved):
            cv2.imwrite(detection_image, frame)
            self.image_saved = True
            self.get_logger().info(f'Saved {detection_image} ({frame.shape[1]}x{frame.shape[0]})')

    def process_image(self):
        '''
        Description:    Timer function: detect, track, publish and display, every cycle.
        '''
        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return
        try:
            frame = self.cv_image.copy()
            self.update_tracks(self.locate_detections(frame))
            self.publish_tracks()
            self.draw_and_save(frame)
        except Exception as exc:                            # never let one bad frame kill the node
            self.get_logger().error(f'process_image: {exc}')


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ore_tf class to perform its task
    '''
    rclpy.init(args=sys.argv)
    node = ore_tf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
