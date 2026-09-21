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

# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		    task1a.py
# Functions:
#			        detect_ores, ore_tf.__init__, ore_tf.depthimagecb, ore_tf.colorimagecb,
#                   ore_tf.caminfocb, ore_tf._on_mouse, ore_tf.process_image, ore_tf._median_depth,
#                   ore_tf._deproject_to_base, ore_tf._match_frame, ore_tf._plausible, main
# Nodes:		    Add your publishing and subscribing node
#                   Example:
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw, /etc... ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image
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

# HSV bounds per ore colour (H: 0-179, S/V: 0-255 in OpenCV). These are STARTING POINTS,
# not measured values - grab a frame with cv2.imwrite while your node is running, sample
# the actual pixels with an eyedropper/cv2.cvtColor, and tighten these to match.
color_bounds = {
    'azurite_ore':    (np.array([100, 180,  150]), np.array([112, 255, 255])),  # blue
    'malachite_ore':  (np.array([50,   140,  50]), np.array([90,  255, 255])),  # green
    'vanadinite_ore': (np.array([5,    120, 120]), np.array([16,  255, 255])),  # orange
    # ^ was S>=180, V>=180 - a real pixel sample off the block that never got
    # published measured S~150, so the old floor rejected genuinely orange, just
    # less-saturated (shaded/angled), pixels rather than a different colour. Given
    # margin here; if it's still missed, click a few more points on that specific
    # block in the detection window and tighten from the actual numbers.
}

min_contour_area_px = 80     # drop blobs smaller than this - tune against a saved frame
match_dist_px = 20           # matching radius for an UNCONFIRMED candidate
# ^ the ores never move once spawned, so this only needs to cover a couple of pixels of
# detection jitter between frames - not how far apart two DIFFERENT ores of the same
# type might land. 60 was too generous: on one real layout it let two distinct malachite
# ores fall within range of each other and both got labelled _1.

confirmed_match_dist_px = 8  # much tighter radius once a track is CONFIRMED - it already
# has an established position, so a false detection landing nearby is far less likely to
# fall this close and steal a frame's publish from the real ore.

# geometric filter on each blob, applied alongside the colour mask - rejects a small
# reflection off the arm's own metal surface, which is usually thin/elongated rather
# than the roughly-square, solidly-filled shape of a cube's top face
min_aspect, max_aspect = 0.55, 1.8    # bounding-box width/height must fall in this range
min_fill_ratio = 0.55                 # contour area / bounding-box area must be at least this

# an id is only trusted (and published) once its candidate has survived this many
# frames net (misses decay the streak rather than resetting it - see _match_frame). A
# one-off reflection that slips past the geometric filter dies out as an unconfirmed
# candidate instead of permanently occupying a real ore's _1/_2 slot. Once confirmed, a
# track is kept for the rest of the run regardless of later misses - the ore is static,
# so there's no legitimate way for a confirmed real detection to vanish, and cycling
# its id would only interrupt an otherwise-correct transform (Task 1A: "a transform
# ... only correct for a moment is not counted").
confirm_hits = 3
provisional_miss_limit = 2      # drop an unconfirmed candidate fast if it stops reappearing

# All 6 ores sit on ONE conveyor, so once a couple are confirmed (of any colour) they
# pin down roughly where "the conveyor" is in 3D. A candidate whose computed position
# is nowhere near that cluster - e.g. a warm reflection on the arm, or on the bin - gets
# refused confirmation even if its colour and shape both look right. This matters
# because a genuinely STATIC false detection (nothing in this scene moves) is just as
# frame-to-frame stable as a real ore, so the hit-streak check alone can't tell them
# apart; this is a starting guess, loosen/tighten it against your own ore layout.
cluster_radius_m = 0.35

# The camera sees the whole cell - arm, storage bin, and conveyor - but every ore
# always sits on the conveyor, which occupies roughly the right portion of the frame
# regardless of where a given ore is randomised to along it. Blanking out everything
# left of this fraction of the frame width removes the arm and the bin from detect_ores
# entirely, rather than relying on colour/geometry/tracking to reject them after the
# fact. A fraction, not a pixel count, so it doesn't depend on the image's resolution.
# Starting guess from the two example frames seen so far - the ROI line drawn on the
# detection window (process_image) shows exactly where this falls; nudge it if the
# conveyor's near edge gets clipped, or if the arm still creeps into it.
roi_x_frac = 0.5

# depth and colour frames arrive on independent callbacks - if they're ever this far
# apart in stamp, something's actually stalled (not just normal per-topic jitter)
max_frame_gap_s = 0.5


##################### FUNCTION DEFINITIONS #######################

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
    roi_x0 = int(image.shape[1] * roi_x_frac)   # everything left of this column is ignored

    for ore_type in ore_types:
        lower, upper = color_bounds[ore_type]
        mask = cv2.inRange(hsv, lower, upper)
        mask[:, :roi_x0] = 0   # arm, bin, background - never even considered
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_contour_area_px:
                continue

            _, _, bw, bh = cv2.boundingRect(c)
            if bw == 0 or bh == 0:
                continue
            aspect = bw / float(bh)
            fill_ratio = area / float(bw * bh)
            if not (min_aspect <= aspect <= max_aspect) or fill_ratio < min_fill_ratio:
                continue   # too thin/sparse to be a cube's top face - likely a reflection

            m = cv2.moments(c)
            if m['m00'] == 0:
                continue
            cX = int(m['m10'] / m['m00'])
            cY = int(m['m01'] / m['m00'])

            center_ore_list.append((cX, cY))
            ore_type_list.append(ore_type)

            # detection boundary - the exact ore name is overlaid later in process_image,
            # once this detection has been matched to an id
            cv2.circle(image, (cX, cY), 18, (0, 255, 0), 2)

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

        super().__init__(                                                               # registering node
            'ore_tf_publisher',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)

        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())
        self.cam_info = None                                                            # camera intrinsics variable (from caminfocb())

        ############ ADD YOUR CODE HERE ############

        self.color_header = None                                                        # header (stamp + optical frame) of the latest colour frame
        self.depth_header = None                                                        # header of the latest depth frame - used to sanity-check the two aren't stale relative to each other
        # keeps azurite_ore_1 vs azurite_ore_2 (etc.) stable for the whole run. Each
        # entry: {'id': 1 or 2, 'px': (cX, cY), 'hits': int, 'misses': int,
        # 'confirmed': bool, 'point': (x, y, z) in base_frame or None} - see
        # _match_frame() for how a candidate earns its id, and process_image() for how
        # 'point' is kept fresh and republished every cycle once confirmed.
        self.ore_tracks = {t: [] for t in ore_types}
        self.confirmed_points = []       # (x, y, z) in base_frame, one per CONFIRMED track of any type

        # click anywhere in the detection window to print that pixel's HSV in the log -
        # the fast way to read real color_bounds off your own frame instead of guessing
        cv2.namedWindow('Task 1A - Ore Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Task 1A - Ore Detection', self._on_mouse)

        ############################################


    def depthimagecb(self, data):
        '''
        Description:    Callback function for the aligned depth camera topic.
                        Use this function to receive the depth image and convert it to a CV2 image.

        Args:
            data (Image):    Input depth image frame received from the aligned depth camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            self.depth_header = data.header
        except CvBridgeError as e:
            self.get_logger().error(f'depthimagecb: {e}')

        ############################################


    def colorimagecb(self, data):
        '''
        Description:    Callback function for the colour camera raw topic.
                        Use this function to receive the raw image and convert it to a CV2 image.

        Args:
            data (Image):    Input coloured raw image frame received from the image_raw camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            self.color_header = data.header
        except CvBridgeError as e:
            self.get_logger().error(f'colorimagecb: {e}')

        ############################################


    def caminfocb(self, data):
        '''
        Description:    Callback function for the camera info topic.
                        Use this function to receive the camera's intrinsic parameters.

        Args:
            data (CameraInfo):    Camera calibration published by the camera

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        self.cam_info = data                                                            # k = [fx, 0, cx, 0, fy, cy, 0, 0, 1], read on demand below

        ############################################


    def _on_mouse(self, event, x, y, flags, param):
        '''Click the detection window: prints the BGR/HSV of that pixel in the current
        frame, so color_bounds can be set from measured values rather than a guess.'''

        if event == cv2.EVENT_LBUTTONDOWN and self.cv_image is not None:
            b, g, r = self.cv_image[y, x]
            hsv_px = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)[y, x]
            self.get_logger().info(
                f'Clicked ({x},{y}): BGR=({int(b)},{int(g)},{int(r)})  HSV={tuple(int(v) for v in hsv_px)}')


    def _median_depth(self, cX, cY, half_win=4):
        '''Robust depth at a pixel: the median of a small window, in METRES, skipping
        invalid (zero / non-finite) readings. Returns None if nothing valid was found.'''

        h, w = self.depth_image.shape[:2]
        y0, y1 = max(cY - half_win, 0), min(cY + half_win + 1, h)
        x0, x1 = max(cX - half_win, 0), min(cX + half_win + 1, w)
        patch = self.depth_image[y0:y1, x0:x1].astype(np.float32)

        if self.depth_image.dtype == np.uint16:
            patch = patch / 1000.0        # 16UC1 is millimetres

        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))


    def _deproject_to_base(self, u, v, z, header):
        '''Back-project pixel (u, v) at depth z (metres) into base_frame. Returns
        (x, y, z) in base_frame, or None if intrinsics or the tf tree are not ready yet.'''

        if self.cam_info is None:
            return None

        fx, fy = self.cam_info.k[0], self.cam_info.k[4]
        cx, cy = self.cam_info.k[2], self.cam_info.k[5]

        point_cam = PointStamped()
        point_cam.header = header
        point_cam.point.x = (u - cx) * z / fx
        point_cam.point.y = (v - cy) * z / fy
        point_cam.point.z = z

        try:
            tf = self.tf_buffer.lookup_transform(base_frame, header.frame_id, Time.from_msg(header.stamp))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            # falls back to nothing this cycle rather than a wrong-time transform;
            # process_image runs again in 0.2s regardless
            self.get_logger().warn(f'TF {header.frame_id} -> {base_frame} not ready: {e}', throttle_duration_sec=2.0)
            return None

        point_base = do_transform_point(point_cam, tf)
        return point_base.point.x, point_base.point.y, point_base.point.z


    def _match_frame(self, ore_type, detections_px):
        '''
        Match this frame's detections (for one ore_type) against this type's tracks.

        Fixes to the earlier one-at-a-time version:
          - Two detections in the SAME frame could both claim the same track (whichever
            was nearest), so two distinct ores could end up publishing under one id.
            Matching is one-to-one: cheapest (track, detection) pair first, and once
            either side is used it's removed from further consideration this frame.
          - A single stray detection could grab an id slot and hold it, blocking the
            real ore's id. A new candidate starts UNCONFIRMED, and only reaching
            'confirm_hits' net hits (misses decay the streak by 1 rather than
            resetting it - see below) makes it ELIGIBLE for confirmation. The actual
            confirm decision happens in process_image, once this frame's 3D point is
            known, so a spatially-implausible candidate can still be refused even after
            it's racked up enough hits. An unconfirmed candidate that stops reappearing
            is dropped fast (provisional_miss_limit).
          - Once CONFIRMED, a track matches on a much tighter radius
            (confirmed_match_dist_px) than an unconfirmed one, so a false detection
            landing near an already-established real ore is far less likely to steal
            its frame. A confirmed track is also never pruned for the rest of the run:
            the ore is static, so there's no legitimate reason for a genuinely confirmed
            detection to disappear, and cycling its id would only interrupt an
            otherwise-correct transform.

        Returns { detection_index: track_dict } for every detection that matched a
        track this frame, confirmed or not - process_image computes each one's 3D
        point, updates confirmed tracks' cached point, and decides confirmation for
        anything that just reached confirm_hits.
        '''

        tracks = self.ore_tracks[ore_type]

        def radius(tr):
            return confirmed_match_dist_px if tr['confirmed'] else match_dist_px

        # one-to-one match: cheapest (track, detection) pairs within range, claimed first
        pairs = []
        for ti, tr in enumerate(tracks):
            for di, px in enumerate(detections_px):
                d = math.hypot(tr['px'][0] - px[0], tr['px'][1] - px[1])
                if d < radius(tr):
                    pairs.append((d, ti, di))
        pairs.sort(key=lambda p: p[0])

        used_tracks, used_dets = set(), set()
        matched = {}
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

        # age out ONLY unconfirmed tracks - a confirmed one is kept, unmodified, for the
        # rest of the run regardless of how many frames it's missed. An unconfirmed
        # candidate's hit streak DECAYS by 1 on a miss rather than resetting to 0: a
        # detector that's marginal (HSV floor right at the edge, an unevenly-lit face)
        # can still confirm as long as hits clearly outnumber misses over a short
        # window, while a genuine one-off false positive - a single hit, then nothing -
        # still decays straight back to 0 and never confirms.
        for ti, tr in enumerate(tracks):
            if ti in used_tracks or tr['confirmed']:
                continue
            tr['misses'] += 1
            tr['hits'] = max(0, tr['hits'] - 1)
        tracks[:] = [tr for tr in tracks if tr['confirmed'] or tr['misses'] <= provisional_miss_limit]

        # an unmatched detection becomes a new, UNCONFIRMED candidate if an id is free
        used_ids = {tr['id'] for tr in tracks}
        for di, px in enumerate(detections_px):
            if di in used_dets or len(tracks) >= 2:
                continue
            free_id = next((i for i in (1, 2) if i not in used_ids), None)
            if free_id is None:
                continue
            tr = {'id': free_id, 'px': px, 'hits': 1, 'misses': 0, 'confirmed': False, 'point': None}
            tracks.append(tr)
            used_ids.add(free_id)
            matched[di] = tr

        return matched


    def _plausible(self, point):
        '''True if 'point' (x, y, z in base_frame) is within cluster_radius_m of ANY
        already-confirmed ore (any colour) - or if there are none yet to compare
        against, in which case the first one or two confirmations of the run are let
        through unchecked, since they're what defines the cluster in the first place.'''

        if not self.confirmed_points:
            return True
        return any(math.dist(point, p) <= cluster_radius_m for p in self.confirmed_points)


    def process_image(self):
        '''
        Description:    Timer function used to detect the ores and publish a transform for
                        each one on its estimated position.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return

        # loose sync guard: independent callbacks don't guarantee the two frames were
        # captured at the same instant. The scene is static in this task so a normal
        # one-tick offset changes nothing - this only catches a topic actually stalling.
        if self.color_header is not None and self.depth_header is not None:
            gap_s = abs((Time.from_msg(self.color_header.stamp) -
                         Time.from_msg(self.depth_header.stamp)).nanoseconds) / 1e9
            if gap_s > max_frame_gap_s:
                self.get_logger().warn(f'colour/depth frames are {gap_s:.2f}s apart - skipping this cycle',
                                        throttle_duration_sec=2.0)
                return

        frame = self.cv_image.copy()
        centers, types = detect_ores(frame)

        by_type = {t: [] for t in ore_types}
        for px, ore_type in zip(centers, types):
            by_type[ore_type].append(px)

        for ore_type, px_list in by_type.items():
            matched = self._match_frame(ore_type, px_list)

            for di, tr in matched.items():
                cX, cY = px_list[di]
                z = self._median_depth(cX, cY)
                if z is None:
                    continue
                result = self._deproject_to_base(cX, cY, z, self.color_header)
                if result is None:
                    continue

                if tr['confirmed']:
                    tr['point'] = result   # just keep the cache fresh
                    continue

                tr['point'] = result       # needed either way, below

                if tr['hits'] >= confirm_hits:
                    if self._plausible(result):
                        tr['confirmed'] = True
                        self.confirmed_points.append(result)
                        self.get_logger().info(f'{ore_type}_{tr["id"]} confirmed at {result}.')
                    else:
                        # colour and shape both looked right, but this is nowhere near
                        # the other ores - almost certainly a static reflection, not a
                        # 7th thing on the conveyor. Refuse it and make it re-earn hits.
                        self.get_logger().warn(
                            f'{ore_type}_{tr["id"]} candidate at {result} is implausibly far '
                            f'from the other ores - not confirming.', throttle_duration_sec=2.0)
                        tr['hits'] = 0

            # publish EVERY confirmed track for this type from its cached point, whether
            # or not it was freshly matched this cycle. The ore doesn't move, so the
            # last good fix stays valid - this keeps the transform broadcasting
            # continuously instead of leaving a gap on a single missed/failed frame.
            for tr in self.ore_tracks[ore_type]:
                if not tr['confirmed'] or tr['point'] is None:
                    continue

                name = f'{ore_type}_{tr["id"]}'
                x_b, y_b, z_b = tr['point']

                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = base_frame
                t.child_frame_id = name
                t.transform.translation.x = float(x_b)
                t.transform.translation.y = float(y_b)
                t.transform.translation.z = float(z_b)
                t.transform.rotation.w = 1.0        # identity - orientation is not scored, but must be a legal quaternion
                self.br.sendTransform(t)

                cX, cY = tr['px']
                cv2.putText(frame, name, (max(cX - 55, 0), max(cY - 22, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # thin marker line at the ROI boundary, purely visual - confirms at a glance
        # whether roi_x_frac is cutting in the right place on this layout
        roi_x0 = int(frame.shape[1] * roi_x_frac)
        cv2.line(frame, (roi_x0, 0), (roi_x0, frame.shape[0]), (255, 255, 0), 1)

        cv2.imshow('Task 1A - Ore Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # quick way to grab the submission screenshot - replace <TEAM_ID> and re-save
            # with the exact name the Submission page asks for before you zip it up
            cv2.imwrite('SC#<TEAM_ID>_task1A_detection.png', frame)
            self.get_logger().info('Saved detection frame (rename with your team ID).')

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the ore_tf
                    class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('ore_tf_process')                      # creating ROS node

    node.get_logger().info('Node created: Ore tf process')          # logging information

    ore_tf_class = ore_tf()                                         # creating a new object for class 'ore_tf'

    rclpy.spin(ore_tf_class)                                        # spining on the object to make it alive in ROS 2 DDS

    ore_tf_class.destroy_node()                                     # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':

    main()
