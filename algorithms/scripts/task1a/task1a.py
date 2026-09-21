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
*****************************************************************************************
'''

# Team ID:          6494
# Author List:      Krishang Nigam, Harshil Jayswal, Masum Pancholi, Divy Vaghasiya
# Filename:         task1a.py
# Functions:
#                     detect_ores, ore_tf.__init__, ore_tf.depthimagecb, ore_tf.colorimagecb,
#                     ore_tf.caminfocb, ore_tf._on_mouse, ore_tf.process_image, ore_tf._median_depth,
#                     ore_tf._deproject_to_base, ore_tf._match_frame, ore_tf._plausible, main
# Nodes:
#                     Publishing Topics  - [ /tf ]
#                     Subscribing Topics - [ /camera/camera/color/image_raw,
#                                           /camera/camera/aligned_depth_to_color/image_raw,
#                                           /camera/camera/color/camera_info ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point


##################### TASK CONSTANTS #######################

# Two ores of each type are spawned - six in all - told apart by an id of 1 or 2.
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

# The RealSense topics. The depth image is ALIGNED to the colour image.
color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

# The parent frame every ore transform is published against.
base_frame = 'base_link'

# HSV bounds measured from the current simulation.
color_bounds = {
    'azurite_ore':
        (np.array([100, 180, 150]),
         np.array([112, 255, 255])),
    'malachite_ore':
        (np.array([50, 140, 50]),
         np.array([90, 255, 255])),
    'vanadinite_ore':
        (np.array([5, 120, 120]),
         np.array([16, 255, 255])),
}

# Small coloured faces can be partly hidden by another ore.
min_contour_area_by_type = {
    'azurite_ore': 20,
    'malachite_ore': 20,
    'vanadinite_ore': 15,
}

# ROI: right/lower region contains the ore conveyor.
# No ROI guide line is drawn in the output window.
roi_x_frac = 0.50
roi_y_min_frac = 0.40
roi_y_max_frac = 0.99

# Keep IDs stable between frames.
match_dist_px = 15
confirmed_match_dist_px = 8

# Confirmation / provisional-track handling.
confirm_hits = 3
provisional_miss_limit = 2

# Observed real ore plane in base_link is around z=-0.065 m.
# This is deliberately wide enough for the partially occluded ore, but rejects
# the false orange regions previously observed around z=-0.55 m.
conveyor_z_min = -0.15
conveyor_z_max = 0.00

# Current working spatial plausibility radius.
# It is only applied after a candidate has reached confirm_hits.
cluster_radius_m = 0.40


##################### FUNCTION DEFINITION #######################

def detect_ores(image):
    '''
    Description:    Function to detect the ores present in a colour image frame and
                    return the pixel location and the type of each one found.

    Args:
        image                   (Image):    Input colour image frame received from the camera topic

    Returns:
        center_ore_list         (list):     Center pixel (cX, cY) of every ore detected in the frame
        ore_type_list           (list):     Type of each ore detected, taken from 'ore_types'
    '''

    ############ Function VARIABLES ############

    center_ore_list = []
    ore_type_list = []

    ############ ADD YOUR CODE HERE ############

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    height, width = image.shape[:2]

    roi_x0 = int(width * roi_x_frac)
    roi_y0 = int(height * roi_y_min_frac)
    roi_y1 = int(height * roi_y_max_frac)

    for ore_type in ore_types:

        lower, upper = color_bounds[ore_type]

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        # Ignore everything outside the conveyor ROI.
        mask[:roi_y0, :] = 0
        mask[roi_y1:, :] = 0
        mask[:, :roi_x0] = 0

        # Mild cleanup.  Do not use aggressive opening because one ore can
        # be partially occluded.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < min_contour_area_by_type[ore_type]:
                continue

            m = cv2.moments(contour)

            if m['m00'] == 0:
                continue

            cX = int(m['m10'] / m['m00'])
            cY = int(m['m01'] / m['m00'])

            center_ore_list.append(
                (cX, cY)
            )

            ore_type_list.append(
                ore_type
            )

    ############################################

    return center_ore_list, ore_type_list


##################### CLASS DEFINITION #######################

class ore_tf(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to detect the ores in the cell and
                    broadcast a transform for each one.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ore_tf
        '''

        super().__init__(
            'ore_tf_publisher',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.Parameter.Type.BOOL,
                    True
                )
            ]
        )

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(
            Image,
            color_topic,
            self.colorimagecb,
            10
        )

        self.depth_cam_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depthimagecb,
            10
        )

        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.caminfocb,
            10
        )

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2

        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.buffer.Buffer()

        self.listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.br = tf2_ros.TransformBroadcaster(
            self
        )

        self.timer = self.create_timer(
            image_processing_rate,
            self.process_image
        )

        self.cv_image = None
        self.depth_image = None
        self.cam_info = None

        ############ ADD YOUR CODE HERE ############

        self.color_header = None
        self.depth_header = None

        # Two stable tracks per ore type.
        # Each track:
        #   id, px, hits, misses, confirmed, point
        self.ore_tracks = {
            ore_type: []
            for ore_type in ore_types
        }

        # Confirmed 3D points used by _plausible().
        self.confirmed_points = []

        # Development helper for measured HSV values.
        cv2.namedWindow(
            'Task 1A - Ore Detection',
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            'Task 1A - Ore Detection',
            640,
            480
        )

        cv2.setMouseCallback(
            'Task 1A - Ore Detection',
            self._on_mouse
        )

        ############################################


    def depthimagecb(self, data):
        '''
        Description:    Callback function for the aligned depth camera topic.
                        Use this function to receive the depth image and convert it to a CV2 image.

        Args:
            data (Image):    Input depth image received from the aligned depth camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                data,
                desired_encoding='passthrough'
            )

            self.depth_header = data.header

        except CvBridgeError as e:
            self.get_logger().error(
                f'depthimagecb: {e}'
            )

        ############################################


    def colorimagecb(self, data):
        '''
        Description:    Callback function for the colour camera raw image.
                        Use this function to receive the colour image and store it.
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(
                data,
                desired_encoding='bgr8'
            )

            self.color_header = data.header

        except CvBridgeError as e:
            self.get_logger().error(
                f'colorimagecb: {e}'
            )

        ############################################


    def caminfocb(self, data):
        '''
        Description:    Callback function for the camera info topic.
                        Store the camera intrinsics; do not hard-code them.
        '''

        ############ ADD YOUR CODE HERE ############

        self.cam_info = data

        ############################################


    def _on_mouse(self, event, x, y, flags, param):
        '''
        Development helper:
        click a pixel in the detection window to print BGR and HSV.
        '''

        if (
            event == cv2.EVENT_LBUTTONDOWN
            and self.cv_image is not None
            and 0 <= y < self.cv_image.shape[0]
            and 0 <= x < self.cv_image.shape[1]
        ):

            b, g, r = self.cv_image[y, x]

            hsv = cv2.cvtColor(
                self.cv_image,
                cv2.COLOR_BGR2HSV
            )

            h, s, v = hsv[y, x]

            self.get_logger().info(
                f'Clicked ({x},{y}): '
                f'BGR=({int(b)},{int(g)},{int(r)}) '
                f'HSV=({int(h)},{int(s)},{int(v)})'
            )


    def _median_depth(self, cX, cY, half_win=4):
        '''
        Description:    Read the median depth in the 9x9 window centred on
                        the detected ore pixel.

        Returns:
            Depth in metres, or None if no valid reading exists.
        '''

        if self.depth_image is None:
            return None

        height, width = self.depth_image.shape[:2]

        x0 = max(
            cX - half_win,
            0
        )

        x1 = min(
            cX + half_win + 1,
            width
        )

        y0 = max(
            cY - half_win,
            0
        )

        y1 = min(
            cY + half_win + 1,
            height
        )

        patch = self.depth_image[
            y0:y1,
            x0:x1
        ].astype(np.float32)

        if self.depth_image.dtype == np.uint16:
            patch = patch / 1000.0

        valid = patch[
            np.isfinite(patch)
            & (patch > 0.0)
        ]

        if valid.size == 0:
            return None

        return float(
            np.median(valid)
        )


    def _deproject_to_base(self, u, v, z, header):
        '''
        Description:    Back-project (u,v,z) into the camera optical frame and
                        transform the point into base_link.

        Returns:
            (x, y, z) in base_link, or None if TF/camera information is unavailable.
        '''

        if (
            self.cam_info is None
            or header is None
        ):
            return None

        fx = float(
            self.cam_info.k[0]
        )

        fy = float(
            self.cam_info.k[4]
        )

        cx = float(
            self.cam_info.k[2]
        )

        cy = float(
            self.cam_info.k[5]
        )

        if fx <= 0.0 or fy <= 0.0:
            return None

        point_camera = PointStamped()

        point_camera.header = header

        point_camera.point.x = (
            (float(u) - cx) * z / fx
        )

        point_camera.point.y = (
            (float(v) - cy) * z / fy
        )

        point_camera.point.z = float(z)

        try:

            tf = self.tf_buffer.lookup_transform(
                base_frame,
                header.frame_id,
                Time()
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ):

            return None

        point_base = do_transform_point(
            point_camera,
            tf
        )

        return (
            float(point_base.point.x),
            float(point_base.point.y),
            float(point_base.point.z)
        )


    def _match_frame(self, ore_type, detections_px):
        '''
        Description:    Match the unordered detections for one ore type against
                        existing tracks using one-to-one nearest-pixel matching.

        Returns:
            Dictionary mapping detection index to matched track.
        '''

        tracks = self.ore_tracks[ore_type]

        # Build all admissible pairs.
        pairs = []

        for track_index, track in enumerate(tracks):

            radius = (
                confirmed_match_dist_px
                if track['confirmed']
                else match_dist_px
            )

            for detection_index, px in enumerate(detections_px):

                distance = math.hypot(
                    track['px'][0] - px[0],
                    track['px'][1] - px[1]
                )

                if distance <= radius:

                    pairs.append(
                        (
                            distance,
                            track_index,
                            detection_index
                        )
                    )

        pairs.sort(
            key=lambda item: item[0]
        )

        used_tracks = set()
        used_detections = set()

        matched = {}

        # Closest pair first, and each track/detection can be used once.
        for _, track_index, detection_index in pairs:

            if track_index in used_tracks:
                continue

            if detection_index in used_detections:
                continue

            track = tracks[track_index]

            track['px'] = detections_px[detection_index]
            track['hits'] += 1
            track['misses'] = 0

            used_tracks.add(
                track_index
            )

            used_detections.add(
                detection_index
            )

            matched[detection_index] = track

        # Age only provisional tracks.
        for track_index, track in enumerate(tracks):

            if track_index in used_tracks:
                continue

            if track['confirmed']:
                continue

            track['misses'] += 1
            track['hits'] = max(
                0,
                track['hits'] - 1
            )

        # Remove stale provisional tracks.
        tracks[:] = [
            track
            for track in tracks
            if (
                track['confirmed']
                or track['misses'] <= provisional_miss_limit
            )
        ]

        # Add remaining detections to free IDs.
        used_ids = {
            track['id']
            for track in tracks
        }

        for detection_index, px in enumerate(detections_px):

            if detection_index in used_detections:
                continue

            if len(tracks) >= 2:
                break

            free_id = next(
                (
                    candidate_id
                    for candidate_id in (1, 2)
                    if candidate_id not in used_ids
                ),
                None
            )

            if free_id is None:
                continue

            track = {
                'id': free_id,
                'px': px,
                'hits': 1,
                'misses': 0,
                'confirmed': False,
                'point': None
            }

            tracks.append(
                track
            )

            used_ids.add(
                free_id
            )

            matched[detection_index] = track

        return matched


    def _plausible(self, point):
        '''
        Description:    Reject a stable colour false-positive if its 3D position
                        is nowhere near the already-confirmed ore group.
        '''

        if not self.confirmed_points:
            return True

        return any(
            math.dist(
                point,
                confirmed_point
            ) <= cluster_radius_m
            for confirmed_point in self.confirmed_points
        )


    def process_image(self):
        '''
        Description:    Timer function used to detect the ores and publish a transform for
                        each one on its estimated position.
        '''

        ############ ADD YOUR CODE HERE ############

        if (
            self.cv_image is None
            or self.depth_image is None
            or self.cam_info is None
        ):
            return

        frame = self.cv_image.copy()

        # ------------------------------------------------------------
        # 1. Find colour detections.
        # ------------------------------------------------------------

        centers, types = detect_ores(
            frame
        )

        valid_by_type = {
            ore_type: []
            for ore_type in ore_types
        }

        # ------------------------------------------------------------
        # 2. Compute a 3D point for every detection and reject only
        #    obviously off-conveyor points.
        # ------------------------------------------------------------

        for (cX, cY), ore_type in zip(
            centers,
            types
        ):

            z = self._median_depth(
                cX,
                cY
            )

            if z is None:
                continue

            result = self._deproject_to_base(
                cX,
                cY,
                z,
                self.color_header
            )

            if result is None:
                continue

            x_b, y_b, z_b = result

            if not (
                conveyor_z_min
                <= z_b
                <= conveyor_z_max
            ):
                continue

            valid_by_type[ore_type].append(
                {
                    'px': (cX, cY),
                    'point': (
                        float(x_b),
                        float(y_b),
                        float(z_b)
                    )
                }
            )

        # ------------------------------------------------------------
        # 3. Match detections to stable IDs.
        # ------------------------------------------------------------

        for ore_type in ore_types:

            detections = valid_by_type[ore_type]

            detections_px = [
                detection['px']
                for detection in detections
            ]

            matched = self._match_frame(
                ore_type,
                detections_px
            )

            # Update the 3D point associated with each matched track.
            for detection_index, track in matched.items():

                point = detections[
                    detection_index
                ]['point']

                track['point'] = point

                # A provisional track needs repeated valid observations.
                if (
                    not track['confirmed']
                    and track['hits'] >= confirm_hits
                ):

                    if self._plausible(point):

                        track['confirmed'] = True

                        self.confirmed_points.append(
                            point
                        )

                        self.get_logger().info(
                            f'{ore_type}_{track["id"]} '
                            f'confirmed at {point}.'
                        )

                    else:
                        # Make the candidate earn its confirmation again.
                        track['hits'] = 0

        # ------------------------------------------------------------
        # 4. Publish every confirmed ore on every cycle.
        # ------------------------------------------------------------

        for ore_type in ore_types:

            for track in self.ore_tracks[ore_type]:

                if (
                    not track['confirmed']
                    or track['point'] is None
                ):
                    continue

                x_b, y_b, z_b = track['point']

                name = (
                    f'{ore_type}_{track["id"]}'
                )

                t = TransformStamped()

                t.header.stamp = (
                    self.get_clock()
                    .now()
                    .to_msg()
                )

                t.header.frame_id = base_frame
                t.child_frame_id = name

                t.transform.translation.x = float(x_b)
                t.transform.translation.y = float(y_b)
                t.transform.translation.z = float(z_b)

                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0

                self.br.sendTransform(
                    t
                )

                # Draw only confirmed ores.
                cX, cY = track['px']

                cv2.circle(
                    frame,
                    (cX, cY),
                    18,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    name,
                    (
                        max(cX + 10, 0),
                        max(cY - 8, 18)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # No ROI guide line is displayed.
        cv2.imshow(
            'Task 1A - Ore Detection',
            frame
        )

        key = cv2.waitKey(1) & 0xFF
        # The frame at this point is the annotated frame used for submission.
        annotated_frame = frame

        cv2.imshow(
            'Task 1A - Ore Detection',
            annotated_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite(
                'SC#6494_task1A_detection.png',
                annotated_frame
            )
            self.get_logger().info(
                'Saved SC#6494_task1A_detection.png'
            )

        if key == ord('q'):
            raise KeyboardInterrupt

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the ore_tf
                    class to perform its task.
    '''

    rclpy.init(
        args=sys.argv
    )

    ore_tf_class = ore_tf()

    try:
        rclpy.spin(
            ore_tf_class
        )

    except KeyboardInterrupt:
        pass

    finally:
        ore_tf_class.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

