#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************************************************
* Team ID:          6494
* Author List:      Krishang Nigam, Harshil Jayswal, Masum Pancholi, Divy Vaghasiya
* Filename:         task2A_perception.py
* Theme:            StrataCobot (SC) -- Task 2A
* Functions:        detect_ores_in_image, ore_perception.__init__,
*                   ore_perception.depthimagecb, ore_perception.colorimagecb,
*                   ore_perception.caminfocb, ore_perception.ore_picked_cb,
*                   ore_perception._median_depth_eroded, ore_perception._deproject_to_base,
*                   ore_perception.process_image, ore_perception.broadcast_transforms,
*                   main
* Nodes:            task2A_perception_node
* Publishing:       /tf (tf2_msgs/TFMessage)
* Subscribing:      /camera/camera/color/image_raw
*                   /camera/camera/aligned_depth_to_color/image_raw
*                   /camera/camera/color/camera_info
*                   /ore_picked (std_msgs/String) [from manipulation node]
*****************************************************************************************
'''

from launch.actions import reset_launch_configurations
from collections import deque
import math
import sys
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


# ==============================================================================
# PERCEPTION TUNABLES & CONSTANTS
# ==============================================================================
ORE_TYPES = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/camera/color/camera_info'
BASE_FRAME = 'base_link'

ROI_X_FRAC = 0.50
ROI_Y_MIN_FRAC = 0.40
ROI_Y_MAX_FRAC = 0.99

CONVEYOR_BELT_X_MIN = -2.0
CONVEYOR_BELT_X_MAX = 1.0
CONVEYOR_BELT_Y_MIN = -2.0
CONVEYOR_BELT_Y_MAX = 1.0
CONVEYOR_BELT_Z_MIN = -0.085
CONVEYOR_BELT_Z_MAX = -0.045

COLOR_BOUNDS = {
    'azurite_ore': (np.array([100, 180, 150], dtype=np.uint8),
                    np.array([112, 255, 255], dtype=np.uint8)),
    'malachite_ore': (np.array([50, 140, 50], dtype=np.uint8),
                      np.array([90, 255, 255], dtype=np.uint8)),
    'vanadinite_ore': (np.array([5, 120, 120], dtype=np.uint8),
                       np.array([16, 255, 255], dtype=np.uint8)),
}

MIN_AREA_BY_TYPE = {'azurite_ore': 20, 'malachite_ore': 20, 'vanadinite_ore': 15}
MAX_CONTOUR_AREA = 100000     # effectively off until we measure real areas
SPLIT_AREA_THRESHOLD = 10**9  # splitting disabled until we measure real areas
MAX_ASPECT_RATIO = 2.5        # partly hidden ores aren't square
MIN_FILL_RATIO = 0.40

# Tracking & Smoothing
TRACK_GATE_M = 0.04           # Nearest-neighbour gating distance in base_link (4 cm)
CONFIRM_HITS = 2              # Consecutive detections before TF publishing begins
MEDIAN_FILTER_WINDOW = 5      # Rolling history length for median smoothing
LIFT_OFF_Z_THRESHOLD = 0.04   # z above CONVEYOR_BELT_Z_MAX considered picked
PROMOTE_HITS = 3
# Processing Rates & Debugging
IMAGE_TIMER_PERIOD = 0.10     # 10 Hz perception loop
BROADCAST_TIMER_PERIOD = 0.05 # 20 Hz TF publishing
DEBUG_LOG = True              # Once-per-second debug logging per ore


def split_touching_rect(rect):
    '''
    Halve a 2x-sized touching ore minAreaRect along its longer side.
    Returns two (center, size, angle) tuples.
    '''
    (cx, cy), (w, h), angle = rect
    if w >= h:
        theta = math.radians(angle)
        dx = math.cos(theta) * (w / 4.0)
        dy = math.sin(theta) * (w / 4.0)
        r1 = ((cx - dx, cy - dy), (w / 2.0, h), angle)
        r2 = ((cx + dx, cy + dy), (w / 2.0, h), angle)
    else:
        theta = math.radians(angle + 90.0)
        dx = math.cos(theta) * (h / 4.0)
        dy = math.sin(theta) * (h / 4.0)
        r1 = ((cx - dx, cy - dy), (w, h / 2.0), angle)
        r2 = ((cx + dx, cy + dy), (w, h / 2.0), angle)
    return r1, r2


class ore_perception(Node):
    '''
    ROS 2 node that detects ores from RealSense feed and broadcasts continuous
    base_link -> <kind>_ore_<n> transforms until each ore is picked.
    '''

    def __init__(self):
        super().__init__(
            'task2A_perception_node',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        self.color_sub = self.create_subscription(Image, COLOR_TOPIC, self.colorimagecb, 10)
        self.depth_sub = self.create_subscription(Image, DEPTH_TOPIC, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.caminfocb, 10)
        self.picked_sub = self.create_subscription(String, '/ore_picked', self.ore_picked_cb, 10)

        self.cv_image = None
        self.depth_image = None
        self.cam_info = None
        self.depth_header = None

        # Persistent tracked ores:
        # {frame_id: {'kind': str, 'pos': [x,y,z], 'history': deque, 'hits': int, 'misses': int,
        #             'picked': bool, 'px': (u,v), 'depth': float}}
        self.tracked_ores = {}
        self.picked_ores = set()
        self.kind_counts = {ot: 0 for ot in ORE_TYPES}
        self.pending=[]
        self.last_warn_time = 0.0
        self.last_debug_log_time = 0.0

        # Timers: perception detection loop and continuous TF broadcaster
        self.process_timer = self.create_timer(IMAGE_TIMER_PERIOD, self.process_image)
        self.broadcast_timer = self.create_timer(BROADCAST_TIMER_PERIOD, self.broadcast_transforms)

        self.get_logger().info('Task 2A Perception node initialized with geometric ROI & square shape validation.')

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'colorimagecb error: {e}')

    def depthimagecb(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            self.depth_header = data.header
        except CvBridgeError as e:
            self.get_logger().error(f'depthimagecb error: {e}')

    def caminfocb(self, data):
        self.cam_info = data

    def ore_picked_cb(self, msg):
        '''Called when manipulation node confirms an ore is picked off conveyor.'''
        ore_name = msg.data.strip()
        self.picked_ores.add(ore_name)
        if ore_name in self.tracked_ores:
            self.tracked_ores[ore_name]['picked'] = True
            self.get_logger().info(f'Ore {ore_name} confirmed picked via topic. TF stopped.')

    def _median_depth_eroded(self, mask_poly, center_px):
        '''
        Compute median depth of valid depth pixels on the eroded top face
        to exclude edges, sides, black border, or the conveyor surface.
        '''
        if self.depth_image is None:
            return None

        h, w = self.depth_image.shape[:2]
        single_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(single_mask, [np.int32(mask_poly)], 255)

        # Erode inner region by 2 iterations with 5x5 kernel
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(single_mask, kernel, iterations=1)

        patch = self.depth_image.astype(np.float32)
        if self.depth_image.dtype == np.uint16:
            patch = patch / 1000.0

        valid_depths = patch[(eroded > 0) & np.isfinite(patch) & (patch > 0.2) & (patch < 2.5)]
        if valid_depths.size >= 4:
            return float(np.median(valid_depths))

        # Fallback: small 5x5 patch around minAreaRect center
        cx, cy = int(round(center_px[0])), int(round(center_px[1]))
        x0, x1 = max(cx - 3, 0), min(cx + 4, w)
        y0, y1 = max(cy - 3, 0), min(cy + 4, h)
        fallback_patch = patch[y0:y1, x0:x1]
        valid_fallback = fallback_patch[np.isfinite(fallback_patch) & (fallback_patch > 0.2) & (fallback_patch < 2.5)]
        if valid_fallback.size > 0:
            return float(np.median(valid_fallback))
        return None

    def _deproject_to_base(self, u, v, z, header):
        '''Deproject pixel + depth into base_link 3D coordinates using tf2.'''
        if self.cam_info is None or header is None or z is None:
            return None

        fx = float(self.cam_info.k[0])
        fy = float(self.cam_info.k[4])
        cx = float(self.cam_info.k[2])
        cy = float(self.cam_info.k[5])

        if fx <= 0.0 or fy <= 0.0:
            return None

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
        '''Perception loop: detect, shape filter, split touching, ROI gate, track.'''
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
        # FIX 1 (b, c, d): Colour, Shape, and Touching-Ore Splitting
        # ------------------------------------------------------------------
        for ore_type in ORE_TYPES:
            lower, upper = COLOR_BOUNDS[ore_type]
            mask = cv2.inRange(hsv, lower, upper)

            # Restrict to conveyor image quadrant
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
                # Check if blob is ~2x expected area (touching ores)
                if area >= SPLIT_AREA_THRESHOLD and aspect_ratio >= 1.4:
                    r1, r2 = split_touching_rect(rect)
                    for sub_r in (r1, r2):
                        poly = cv2.boxPoints(sub_r)
                        sub_w, sub_h = sub_r[1]
                        sub_ar = max(sub_w, sub_h) / max(min(sub_w, sub_h), 1e-5)
                        raw_candidates.append({
                            'kind': ore_type,
                            'center_px': sub_r[0],
                            'poly': poly,
                            'area': area / 2.0,
                            'aspect_ratio': sub_ar,
                            'fill_ratio': fill_ratio,
                        })
                elif area <= MAX_CONTOUR_AREA:
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
        # FIX 1 (a) & FIX 3 (a, b, c): Deproject & Filter with Fixed Belt ROI
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

            # Strict Fixed Scene Geometry Acceptance Gate
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
        # FIX 1 (e): Count Safety Net (Exactly 4 Ores Total)
        # ------------------------------------------------------------------
        if len(roi_candidates) > 4:
            # Score candidates: lower score = more square, solid, and close to nominal ore area
            def candidate_quality(c):
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
        # FIX 2: Stable Name Tracking by Nearest-Neighbour in base_link
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

                d = float(np.linalg.norm(cand['pos_base'] - track['pos']))
                if d <= TRACK_GATE_M and d < best_dist:
                    best_dist = d
                    best_track_name = name

            if best_track_name is not None:
                matched_track_names.add(best_track_name)
                track = self.tracked_ores[best_track_name]
                track['history'].append(cand['pos_base'])
                # FIX 3 (d): Median-smooth over the last 3-5 frames
                track['pos'] = np.median(np.array(track['history']), axis=0)
                track['hits'] += 1
                track['misses'] = 0
                track['px'] = cand['center_px']
                track['depth'] = cand['depth']
            else:
                unmatched_candidates.append(cand)

        # Unmatched candidates: sort deterministically by base_link y, then x
        unmatched_candidates.sort(key=lambda c: (c['pos_base'][1], c['pos_base'][0]))

        for p in self.pending:
            p['seen'] = False
        for cand in unmatched_candidates:
            best, bd = None, 1e9
            for p in self.pending:
                if p['kind'] != cand['kind'] or p['seen']:
                    continue
                d = float(np.linalg.norm(cand['pos_base'] - p['pos']))
                if d <= TRACK_GATE_M and d < bd:
                    best, bd = p, d
            if best is not None:
                best.update(seen=True, pos=cand['pos_base'], cand=cand)
                best['hits'] += 1
            else:
                self.pending.append({'kind': cand['kind'], 'pos': cand['pos_base'], 'cand': cand, 'hits': 1, 'seen': True})
        self.pending = [p for p in self.pending if p['seen']]

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
            self.tracked_ores[new_name] = {'kind': kind, 'pos': cand['pos_base'].copy(), 'history': hist,'hits': CONFIRM_HITS, 'misses': 0, 'picked': False,'px': cand['center_px'], 'depth': cand['depth'],
            }

        # Check existing tracks for missed detections or lift-off
        for name, track in self.tracked_ores.items():
            if track['picked']:
                continue
            if name not in matched_track_names:
                track['misses'] += 1
            # Check if ore has lifted off the conveyor
            if track['pos'][2] > (CONVEYOR_BELT_Z_MAX + LIFT_OFF_Z_THRESHOLD):
                track['picked'] = True
                self.get_logger().info(f'{name} lifted off conveyor (z={track["pos"][2]:.3f}m). TF stopped.')

        # ------------------------------------------------------------------
        # FIX 3 (e): Debug Logging (Once per second per ore, no spam)
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
        '''Continuously broadcast TF for all confirmed unpicked ores.'''
        now = self.get_clock().now().to_msg()
        for name, data in self.tracked_ores.items():
            if data['picked'] or name in self.picked_ores:
                continue
            if data['hits'] < CONFIRM_HITS:
                continue

            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = BASE_FRAME
            t.child_frame_id = name
            t.transform.translation.x = float(data['pos'][0])
            t.transform.translation.y = float(data['pos'][1])
            t.transform.translation.z = float(data['pos'][2])
            # Identity rotation for point-centre transform
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = ore_perception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
