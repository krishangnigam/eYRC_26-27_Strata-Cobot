#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************************************************
* Team ID:          6494
* Author List:      Krishang Nigam, Harshil Jayswal, Masum Pancholi, Divy Vaghasiya
* Filename:         task2A_perception.py
* Theme:            StrataCobot (SC) -- Task 2A
* Functions:        detect_ores, ore_perception.__init__, ore_perception.depthimagecb,
*                   ore_perception.colorimagecb, ore_perception.caminfocb,
*                   ore_perception.ore_picked_cb, ore_perception._median_depth,
*                   ore_perception._deproject_to_base, ore_perception.process_image,
*                   ore_perception.broadcast_transforms, main
* Nodes:            task2A_perception_node
* Publishing:       /tf (tf2_msgs/TFMessage)
* Subscribing:      /camera/camera/color/image_raw
*                   /camera/camera/aligned_depth_to_color/image_raw
*                   /camera/camera/color/camera_info
*                   /ore_picked (std_msgs/String) [from manipulation node]
*****************************************************************************************
'''

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

# HSV color ranges for segmentation on conveyor belt
COLOR_BOUNDS = {
    'azurite_ore': (
        np.array([100, 180, 150], dtype=np.uint8),
        np.array([112, 255, 255], dtype=np.uint8),
    ),
    'malachite_ore': (
        np.array([50, 140, 50], dtype=np.uint8),
        np.array([90, 255, 255], dtype=np.uint8),
    ),
    'vanadinite_ore': (
        np.array([5, 120, 120], dtype=np.uint8),
        np.array([16, 255, 255], dtype=np.uint8),
    ),
}

# Minimum contour area per ore kind
MIN_CONTOUR_AREA = {
    'azurite_ore': 20,
    'malachite_ore': 20,
    'vanadinite_ore': 15,
}

# Conveyor region of interest in image fraction
ROI_X_FRAC = 0.50
ROI_Y_MIN_FRAC = 0.40
ROI_Y_MAX_FRAC = 0.99

# Conveyor z-range in base_link (metres) to reject spurious detections
CONVEYOR_Z_MIN = -0.15
CONVEYOR_Z_MAX = 0.05

# Smoothing and tracking parameters
CONFIRM_HITS = 3
MATCH_DIST_PX = 20
ALPHA_SMOOTH = 0.40  # Exponential moving average factor for 3D coordinates

# Processing rates
IMAGE_TIMER_PERIOD = 0.10      # 10 Hz image perception
BROADCAST_TIMER_PERIOD = 0.05  # 20 Hz continuous TF publishing


def detect_ores(image):
    '''
    Detect ores in BGR image using HSV thresholding and contour moments.
    Returns list of ((cX, cY), ore_type).
    '''
    detections = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    height, width = image.shape[:2]
    roi_x0 = int(width * ROI_X_FRAC)
    roi_y0 = int(height * ROI_Y_MIN_FRAC)
    roi_y1 = int(height * ROI_Y_MAX_FRAC)

    for ore_type in ORE_TYPES:
        lower, upper = COLOR_BOUNDS[ore_type]
        mask = cv2.inRange(hsv, lower, upper)

        # Apply ROI
        mask[:roi_y0, :] = 0
        mask[roi_y1:, :] = 0
        mask[:, :roi_x0] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA[ore_type]:
                continue

            m = cv2.moments(contour)
            if m['m00'] <= 0:
                continue

            cX = int(m['m10'] / m['m00'])
            cY = int(m['m01'] / m['m00'])
            detections.append(((cX, cY), ore_type))

    return detections


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

        # Confirmed tracked ores: {frame_id: {'pos': [x, y, z], 'hits': n, 'picked': bool, 'px': (u, v)}}
        self.tracked_ores = {}
        self.picked_ores = set()

        # Timers: perception detection loop and continuous TF broadcaster
        self.process_timer = self.create_timer(IMAGE_TIMER_PERIOD, self.process_image)
        self.broadcast_timer = self.create_timer(BROADCAST_TIMER_PERIOD, self.broadcast_transforms)

        self.get_logger().info('Task 2A Perception node initialized.')

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
            self.get_logger().info(f'Ore {ore_name} marked picked. Ceasing TF publishing.')

    def _median_depth(self, cX, cY, half_win=4):
        '''Sample median depth around (cX, cY) in metres.'''
        if self.depth_image is None:
            return None

        h, w = self.depth_image.shape[:2]
        x0 = max(cX - half_win, 0)
        x1 = min(cX + half_win + 1, w)
        y0 = max(cY - half_win, 0)
        y1 = min(cY + half_win + 1, h)

        patch = self.depth_image[y0:y1, x0:x1].astype(np.float32)
        if self.depth_image.dtype == np.uint16:
            patch = patch / 1000.0

        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _deproject_to_base(self, u, v, z, header):
        '''Deproject pixel + depth into base_link 3D coordinates.'''
        if self.cam_info is None or header is None:
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
        '''Perception loop: detect ores, match to stable names, filter positions.'''
        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return

        detections = detect_ores(self.cv_image)
        # Group detections by ore type
        by_type = {ot: [] for ot in ORE_TYPES}
        for (u, v), ot in detections:
            depth = self._median_depth(u, v)
            if depth is None:
                continue
            pt_base = self._deproject_to_base(u, v, depth, self.depth_header)
            if pt_base is None:
                continue
            bx, by, bz = pt_base
            # Check within spatial conveyor boundary
            if not (CONVEYOR_Z_MIN <= bz <= CONVEYOR_Z_MAX):
                continue
            by_type[ot].append({'px': (u, v), 'pos': [bx, by, bz]})

        # Assign stable names per ore kind
        for ot, items in by_type.items():
            if not items:
                continue

            # Sort items consistently by conveyor coordinate (e.g. descending Y or ascending X)
            # Consistent ordering guarantees identical frame assignment across frames
            items.sort(key=lambda it: it['pos'][1])

            for idx, item in enumerate(items):
                frame_name = f'{ot}_{idx + 1}'
                if frame_name in self.picked_ores:
                    continue

                if frame_name not in self.tracked_ores:
                    self.tracked_ores[frame_name] = {
                        'pos': list(item['pos']),
                        'hits': 1,
                        'picked': False,
                        'px': item['px'],
                    }
                else:
                    track = self.tracked_ores[frame_name]
                    if track['picked']:
                        continue
                    # Exponential smoothing
                    for i in range(3):
                        track['pos'][i] = (1.0 - ALPHA_SMOOTH) * track['pos'][i] + ALPHA_SMOOTH * item['pos'][i]
                    track['hits'] += 1
                    track['px'] = item['px']

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
