#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 3B of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task3B_navigation.py
# Functions:
#                   wrap_angle, yaw_from_quat, clamp, main
# Nodes:            Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom, /scan ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from collections import deque                                           # recent odometry speeds


##################### TASK CONSTANTS #######################

# The planner publishes a path here for each of the three legs, held for late subscribers.
path_topic = '/ebot_path'

odom_topic = '/odom'
scan_topic = '/scan'
cmd_topic = '/cmd_vel'

# What /cmd_vel accepts. The base clips anything beyond these, and clipping one of the
# two changes their ratio - which is the arc the base actually drives.
max_linear_mps = 0.5
max_angular_rps = 1.0


##################### CONTROLLER TUNING #######################

cruise_speed = 0.45             # m/s on straights
min_speed = 0.10                # m/s floor while approaching a stop point
k_slow = 2.0                    # 1/s: speed allowed per metre left to the next stop point
lookahead = 0.30                # m: carrot distance, never past the next stop point
k_heading = 2.0                 # angular gain while driving
k_turn = 3.0                    # angular gain while turning on the spot
min_turn_rate = 0.18            # rad/s, so slip cannot stall a small final correction
max_turn_rate = 1.0             # rad/s while turning on the spot
turn_in_place = 0.45            # rad: heading error above this stops forward motion
corner_angle = 0.25             # rad: a bend sharper than this is taken stopped
align_tol = 0.05                # rad: done aligning to a segment
final_yaw_tol = 0.04            # rad: final heading (the evaluator allows 0.15)
final_pos_tol = 0.03            # m: final position (the evaluator allows 0.3)
settled_yaw_tol = 0.07          # rad: heading accepted once the eBot has come to rest
settled_pos_tol = 0.06          # m: position accepted once the eBot has come to rest
rest_speed = 0.02               # m/s and rad/s: below both, the eBot is at rest
rest_window_s = 0.4             # s of odometry averaged for that check, so noise cannot hide a standstill
max_corrections = 3             # corrections of the final pose before it is accepted as it is
rest_timeout_s = 2.0            # s after which a commanded stop counts as at rest, whatever the speeds read

# The eBot does not turn about its odom point. On the spot it turns about a pivot that lies
# pivot_offset behind that point (measured from recorded runs), so the odom point swings on
# an arc. The pivot is therefore what is steered along the path: it stops on each bend, the
# eBot turns about it, and the odom point comes out on the next segment.
pivot_offset = 0.10             # m from the odom point back to the pivot

# While turning, the pivot is held on its spot by driving along the eBot's own axis.
k_hold = 1.5                    # 1/s: forward/back speed per metre of drift along the axis
max_hold_speed = 0.08           # m/s

# Each segment is driven forwards or backwards, whichever plan needs the least turning on
# the spot over the whole path. Turns at the first and last pose cost more than turns at a
# bend, because those poses are the tight ones and bends are planned with room to turn.
start_turn_weight = 1.2
end_turn_weight = 1.5
stop_penalty = 0.3              # rad: cost added for every stop, on top of the turn itself
aligned_tol = 0.2               # rad: the last segment counts as aligned with the final yaw

min_segment_length = 1e-4       # m: shorter path segments are dropped
merge_distance = 0.06           # m: two bends closer than this are driven as one
segment_end_slack = 5e-3        # m: a segment counts as done this close to its end


##################### MATH UTILITIES #######################

def wrap_angle(angle):
    '''
    Description:    Wraps an angle into [-pi, pi).

    Args:
        angle           (float):    angle in radians

    Returns:
        wrapped         (float):    the same angle, in [-pi, pi)
    '''
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(quat):
    '''
    Description:    Yaw of a geometry_msgs Quaternion.

    Args:
        quat            (Quaternion):   orientation to read

    Returns:
        yaw             (float):    rotation about z, in radians
    '''
    return math.atan2(2.0 * (quat.w * quat.z + quat.x * quat.y), 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z))


def clamp(value, low, high):
    '''
    Description:    Limits a value to [low, high].

    Args:
        value           (float):    number to limit
        low             (float):    smallest allowed value
        high            (float):    largest allowed value

    Returns:
        limited         (float):    'value' kept inside [low, high]
    '''
    return max(low, min(high, value))


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along every path the
                    planner publishes on /ebot_path: turn onto the first segment, track each
                    segment, stop and turn at sharp bends, then stop on the last pose and
                    turn to its yaw.
    '''

    # State machine definitions
    STATE_IDLE = 0              # no path to drive: hold still
    STATE_ALIGN = 1             # turn on the spot onto the current segment
    STATE_FOLLOW = 2            # drive along the segments
    STATE_FINAL_TURN = 3        # at the end of the path: turn to the last pose's yaw
    STATE_FINAL_SHIFT = 4       # move along the final heading onto the last pose
    STATE_SETTLE = 5            # wait until at rest, then check the final pose once more

    def __init__(self):
        '''
        Description:    Initialization of class ebot_nav
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # Each path is published once and held, so the subscription must match: TRANSIENT_LOCAL.
        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############

        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scancb, qos_profile_sensor_data)

        ############ Topic PUBLISHERS ############

        # the eBot drives on what is published here
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05                                                  # seconds per control cycle (20 Hz)
        self.timer = self.create_timer(control_rate, self.process_navigation)

        self.route = None                                                    # the route to drive (from pathcb())
        self.odom = None                                                     # where the base is (from odomcb())
        self.scan = None                                                     # what the lidar sees (from scancb())

        ############ ADD YOUR CODE HERE ############

        self.x = self.y = self.yaw = 0.0
        self.final_yaw = 0.0            # yaw of the path's last pose
        self.last_key = None            # identifies the path already taken
        self.seg = 0                    # index of the segment being tracked
        self.backwards = []             # per segment: True if it is driven backwards
        self.hold_xy = None             # where the pivot is held while turning on the spot
        self.recent_speeds = deque()    # (time, linear, angular) over the last rest_window_s
        self.corrections = 0            # corrections of the final pose made on this path
        self.settle_since = None        # when the eBot was last told to stop on the final pose
        self.state = self.STATE_IDLE

        ############################################


    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
                        Use this function to receive the waypoints the eBot has to drive.

        Args:
            data            (Path):     The route, as a sequence of poses

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # A new path arrives for every step; each one is taken once.
        key = (data.header.stamp.sec, data.header.stamp.nanosec, len(data.poses))
        if len(data.poses) < 2 or key == self.last_key:
            return
        self.last_key = key
        route = [(data.poses[0].pose.position.x, data.poses[0].pose.position.y)]
        for pose_stamped in data.poses[1:]:
            point = (pose_stamped.pose.position.x, pose_stamped.pose.position.y)
            if math.hypot(point[0] - route[-1][0], point[1] - route[-1][1]) > min_segment_length:
                route.append(point)
        if len(route) < 2:
            return
        self.route = self.merge_collinear(route)
        self.final_yaw = yaw_from_quat(data.poses[-1].pose.orientation)
        self.backwards = self.plan_directions()
        self.seg = 0
        self.hold_xy = None
        self.corrections = 0
        self.state = self.STATE_ALIGN
        self.get_logger().info(f'New path: {len(self.route)} vertices in frame {data.header.frame_id}, ends at '
                               f'({route[-1][0]:.2f}, {route[-1][1]:.2f}) yaw {self.final_yaw:.2f}')

        ############################################


    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Use this function to receive where the base currently is.

        Args:
            data            (Odometry): Pose and velocity of the base

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.yaw = yaw_from_quat(data.pose.pose.orientation)
        self.odom = data

        # Keep the last rest_window_s of speeds: their mean is what at_rest() checks
        now = self.get_clock().now().nanoseconds * 1e-9
        self.recent_speeds.append((now, data.twist.twist.linear.x, data.twist.twist.angular.z))
        while self.recent_speeds and now - self.recent_speeds[0][0] > rest_window_s:
            self.recent_speeds.popleft()

        ############################################


    def scancb(self, data):
        '''
        Description:    Callback function for the lidar topic.
                        Use this function to receive what the lidar currently sees.

        Args:
            data            (LaserScan):    One lidar sweep

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # Stored for debugging only. In Task 3B the eBot must follow the path it published,
        # so it never steers off it for a rock seen here; the planned path clears the rocks.
        self.scan = data

        ############################################


    def send(self, linear, angular):
        '''
        Description:    Publishes a Twist, scaled down as a whole if over the limits so the
                        linear to angular ratio - the arc driven - is kept.

        Args:
            linear          (float):    forward speed in m/s (negative = backwards)
            angular         (float):    turn rate in rad/s

        Returns:
        '''
        scale = min(1.0, max_linear_mps / abs(linear) if linear else 1.0,
                    max_angular_rps / abs(angular) if angular else 1.0)
        cmd = Twist()
        cmd.linear.x = float(linear * scale)
        cmd.angular.z = float(angular * scale)
        self.cmd_pub.publish(cmd)


    def merge_collinear(self, route):
        '''
        Description:    Drops the points that lie on a straight stretch, so the route keeps only
                        its start, its bends and its end, and joins bends that nearly coincide.

        Args:
            route           (list):     [(x, y), ...] every pose of the path

        Returns:
            vertices        (list):     [(x, y), ...] the corner points of the path
        '''
        vertices = [route[0]]
        for i in range(1, len(route) - 1):
            before = math.atan2(route[i][1] - vertices[-1][1], route[i][0] - vertices[-1][0])
            after = math.atan2(route[i + 1][1] - route[i][1], route[i + 1][0] - route[i][0])
            if abs(wrap_angle(after - before)) > 1e-3:
                vertices.append(route[i])
        vertices.append(route[-1])

        # Two bends a few centimetres apart would mean two turns on the spot for nothing:
        # they are driven as one bend, at the point midway between them.
        i = 1
        while i < len(vertices) - 2:
            if math.dist(vertices[i], vertices[i + 1]) < merge_distance:
                vertices[i] = ((vertices[i][0] + vertices[i + 1][0]) / 2.0,
                               (vertices[i][1] + vertices[i + 1][1]) / 2.0)
                del vertices[i + 1]
            else:
                i += 1
        return vertices


    def pivot(self):
        '''
        Description:    The point the eBot turns about: pivot_offset behind its odom point.

        Args:

        Returns:
            point           (tuple):    (x, y) of the pivot in the map frame
        '''
        return (self.x - pivot_offset * math.cos(self.yaw), self.y - pivot_offset * math.sin(self.yaw))


    def seg_heading(self, i):
        '''
        Description:    Direction of a segment of the route.

        Args:
            i               (int):      index of the segment (from route[i] to route[i + 1])

        Returns:
            heading         (float):    direction of the segment, in radians
        '''
        (start_x, start_y), (end_x, end_y) = self.route[i], self.route[i + 1]
        return math.atan2(end_y - start_y, end_x - start_x)


    def body_heading(self, i):
        '''
        Description:    The yaw the eBot holds on a segment: along it, or the opposite when the
                        segment is driven backwards.

        Args:
            i               (int):      index of the segment

        Returns:
            heading         (float):    yaw of the eBot on that segment, in radians
        '''
        return self.seg_heading(i) + (math.pi if self.backwards[i] else 0.0)


    def plan_directions(self):
        '''
        Description:    Chooses forwards or backwards for every segment so that the whole path
                        needs the least turning on the spot, from the yaw the eBot has now to
                        the yaw of the last pose.

        Args:

        Returns:
            backwards       (list):     per segment, True if it is driven backwards
        '''
        count = len(self.route) - 1
        headings = [self.seg_heading(i) for i in range(count)]

        def turn_cost(angle, weight=1.0):
            '''
            Description:    Cost of changing yaw by 'angle': nothing for a bend shallow enough
                            to drive through, otherwise the turn plus the stop it needs.

            Args:
                angle           (float):    change of yaw in radians
                weight          (float):    multiplier for turns at the first or last pose

            Returns:
                cost            (float):    cost of that turn
            '''
            angle = abs(wrap_angle(angle))
            return 0.0 if angle <= corner_angle else weight * (angle + stop_penalty)

        # best[d] = (cost so far, choices so far) with the current segment driven forwards (0)
        # or backwards (1)
        best = [(turn_cost(headings[0] + flip * math.pi - self.yaw, start_turn_weight), [bool(flip)])
                for flip in (0, 1)]
        for i in range(1, count):
            step = []
            for flip in (0, 1):
                options = [(best[prev][0] + turn_cost(headings[i] + flip * math.pi
                                                      - headings[i - 1] - prev * math.pi),
                            best[prev][1] + [bool(flip)]) for prev in (0, 1)]
                step.append(min(options, key=lambda option: option[0]))
            best = step
        final = [(best[flip][0] + turn_cost(self.final_yaw - headings[-1] - flip * math.pi, end_turn_weight),
                  best[flip][1]) for flip in (0, 1)]
        return min(final, key=lambda option: option[0])[1]


    def stop_after(self, i):
        '''
        Description:    True when the eBot must stop at the end of segment i: the last
                        segment, or a change of yaw sharper than corner_angle.

        Args:
            i               (int):      index of the segment

        Returns:
            stop            (bool):     True if the eBot stops at the end of that segment
        '''
        return i >= len(self.route) - 2 or \
            abs(wrap_angle(self.body_heading(i + 1) - self.body_heading(i))) > corner_angle


    def stop_length(self, i):
        '''
        Description:    How far along segment i the pivot travels. On the last segment, when the
                        eBot arrives already on the final yaw, the pivot stops where the odom
                        point is on the last pose; everywhere else it stops on the vertex.

        Args:
            i               (int):      index of the segment

        Returns:
            length          (float):    metres from the segment's start to the pivot's stop
        '''
        length = math.dist(self.route[i], self.route[i + 1])
        if i == len(self.route) - 2:
            off_final = wrap_angle(self.final_yaw - self.seg_heading(i))
            if abs(math.sin(off_final)) < math.sin(aligned_tol):
                length -= pivot_offset * math.cos(off_final)
        return length


    def dist_to_next_stop(self, i, along):
        '''
        Description:    Distance the pivot has left along the route to its next stop.

        Args:
            i               (int):      index of the segment the eBot is on
            along           (float):    metres the pivot has covered along that segment

        Returns:
            distance        (float):    metres to the next stop point (never negative)
        '''
        remaining = -along
        while not self.stop_after(i):
            remaining += math.dist(self.route[i], self.route[i + 1])
            i += 1
        return max(remaining + self.stop_length(i), 0.0)


    def carrot(self, i, along, dist):
        '''
        Description:    The point 'dist' metres further along the route from 'along' metres
                        into segment i. It never turns past the next stop point, so bends are
                        never cut; near a stop point it runs on along the same straight line,
                        so the heading stays steady right up to the stop.

        Args:
            i               (int):      index of the segment the eBot is on
            along           (float):    metres the pivot has covered along that segment
            dist            (float):    look-ahead distance in metres

        Returns:
            point           (tuple):    (x, y) of the carrot in the map frame
        '''
        (start_x, start_y), (end_x, end_y) = self.route[i], self.route[i + 1]
        seg_len = math.hypot(end_x - start_x, end_y - start_y)
        target = along + dist
        while target > seg_len and not self.stop_after(i):
            target -= seg_len
            i += 1
            (start_x, start_y), (end_x, end_y) = self.route[i], self.route[i + 1]
            seg_len = math.hypot(end_x - start_x, end_y - start_y)
        fraction = target / seg_len
        return (start_x + (end_x - start_x) * fraction, start_y + (end_y - start_y) * fraction)


    def turn_towards(self, target_yaw, tol):
        '''
        Description:    Turns on the spot towards target_yaw, closing the loop on /odom, while
                        driving along its own axis to keep the pivot on the hold point.

        Args:
            target_yaw      (float):    heading to reach, in radians
            tol             (float):    done within this many radians

        Returns:
            aligned         (bool):     True once within 'tol' (and stopped)
        '''
        err = wrap_angle(target_yaw - self.yaw)
        if abs(err) < tol:
            self.send(0.0, 0.0)
            return True
        turn_rate = clamp(k_turn * err, -max_turn_rate, max_turn_rate)
        turn_rate = turn_rate if abs(turn_rate) >= min_turn_rate else math.copysign(min_turn_rate, err)
        pivot_x, pivot_y = self.pivot()
        along = (self.hold_xy[0] - pivot_x) * math.cos(self.yaw) + (self.hold_xy[1] - pivot_y) * math.sin(self.yaw)
        self.send(clamp(k_hold * along, -max_hold_speed, max_hold_speed), turn_rate)
        return False


    def follow_segment(self):
        '''
        Description:    One control cycle along the current segment: pure pursuit of a carrot
                        that never passes the next stop point, slowing towards it.

        Args:

        Returns:
        '''
        i = self.seg
        (start_x, start_y), (end_x, end_y) = self.route[i], self.route[i + 1]
        seg_len = math.hypot(end_x - start_x, end_y - start_y)
        unit_x, unit_y = (end_x - start_x) / seg_len, (end_y - start_y) / seg_len
        pivot_x, pivot_y = self.pivot()
        along = (pivot_x - start_x) * unit_x + (pivot_y - start_y) * unit_y

        # End of the segment: finish, carry straight on, or stop and turn at a sharp bend.
        if along >= self.stop_length(i) - segment_end_slack:
            if i >= len(self.route) - 2:
                self.send(0.0, 0.0)
                self.hold_xy = (pivot_x, pivot_y)
                self.state = self.STATE_FINAL_TURN
            else:
                sharp = self.stop_after(i)
                self.seg += 1
                if sharp:
                    self.send(0.0, 0.0)
                    self.hold_xy = (end_x, end_y)                 # turn about the bend itself
                    self.state = self.STATE_ALIGN
            return

        reverse = self.backwards[i]
        carrot_x, carrot_y = self.carrot(i, along, lookahead)
        travel_heading = self.yaw + (math.pi if reverse else 0.0)
        err = wrap_angle(math.atan2(carrot_y - pivot_y, carrot_x - pivot_x) - travel_heading)
        to_stop = self.dist_to_next_stop(i, along)

        if abs(err) > turn_in_place and to_stop > final_pos_tol:            # badly off heading: turn first
            self.send(0.0, clamp(k_turn * err, -max_turn_rate, max_turn_rate))
            return

        speed = max(min(cruise_speed, k_slow * to_stop), min_speed) * max(0.0, math.cos(err))
        self.send(-speed if reverse else speed, clamp(k_heading * err, -max_angular_rps, max_angular_rps))


    def at_rest(self):
        '''
        Description:    True when the eBot stands still, judged on the mean of the last
                        rest_window_s of odometry so that sensor noise does not read as motion.

        Args:

        Returns:
            resting         (bool):     True if both mean speeds are below rest_speed
        '''
        if not self.recent_speeds:
            return False
        count = len(self.recent_speeds)
        mean_linear = sum(sample[1] for sample in self.recent_speeds) / count
        mean_angular = sum(sample[2] for sample in self.recent_speeds) / count
        return abs(mean_linear) < rest_speed and abs(mean_angular) < rest_speed


    def final_shift(self):
        '''
        Description:    After the final turn, moves straight along the final heading until the
                        odom point is level with the last pose.

        Args:

        Returns:
            done            (bool):     True once on the last pose (and stopped)
        '''
        goal_x, goal_y = self.route[-1]
        along = (goal_x - self.x) * math.cos(self.yaw) + (goal_y - self.y) * math.sin(self.yaw)
        if abs(along) < final_pos_tol:
            self.send(0.0, 0.0)
            return True
        speed = math.copysign(max(min(k_slow * abs(along), cruise_speed), min_speed), along)
        self.send(speed, clamp(k_heading * wrap_angle(self.final_yaw - self.yaw), -max_turn_rate, max_turn_rate))
        return False


    def process_navigation(self):
        '''
        Description:    Timer function used to drive the eBot along the route.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            # Nothing to drive yet, or the step is done: hold still. Zero velocity on every
            # cycle is what "standing still" means to the planner and the evaluator.
            if self.odom is None or self.route is None or self.state == self.STATE_IDLE:
                self.send(0.0, 0.0)

            # ------------------------------------------------------------------
            # STATE 1: Turn on the spot onto the current segment
            # ------------------------------------------------------------------
            elif self.state == self.STATE_ALIGN:
                if self.hold_xy is None:
                    self.hold_xy = self.pivot()
                if self.turn_towards(self.body_heading(self.seg), align_tol):
                    self.state = self.STATE_FOLLOW

            # ------------------------------------------------------------------
            # STATE 2: Drive along the segments
            # ------------------------------------------------------------------
            elif self.state == self.STATE_FOLLOW:
                self.follow_segment()

            # ------------------------------------------------------------------
            # STATE 3: At the end of the path, turn to the last pose's yaw
            # ------------------------------------------------------------------
            elif self.state == self.STATE_FINAL_TURN:
                if self.turn_towards(self.final_yaw, final_yaw_tol):
                    self.state = self.STATE_FINAL_SHIFT

            # ------------------------------------------------------------------
            # STATE 4: Move along the final heading onto the last pose, then hold still
            # ------------------------------------------------------------------
            elif self.state == self.STATE_FINAL_SHIFT:
                if self.final_shift():
                    self.settle_since = self.get_clock().now().nanoseconds * 1e-9
                    self.state = self.STATE_SETTLE

            # ------------------------------------------------------------------
            # STATE 5: Once at rest, accept the pose or correct it again
            # ------------------------------------------------------------------
            elif self.state == self.STATE_SETTLE:
                self.send(0.0, 0.0)
                waited = self.get_clock().now().nanoseconds * 1e-9 - self.settle_since
                if self.at_rest() or waited >= rest_timeout_s:
                    goal_x, goal_y = self.route[-1]
                    along = (goal_x - self.x) * math.cos(self.yaw) + (goal_y - self.y) * math.sin(self.yaw)
                    retry = self.corrections < max_corrections
                    if retry and abs(wrap_angle(self.final_yaw - self.yaw)) > settled_yaw_tol:
                        self.corrections += 1
                        self.hold_xy = self.pivot()
                        self.state = self.STATE_FINAL_TURN
                    elif retry and abs(along) > settled_pos_tol:
                        self.corrections += 1
                        self.state = self.STATE_FINAL_SHIFT
                    else:
                        self.get_logger().info(f'Arrived at ({self.x:.3f}, {self.y:.3f}) yaw {self.yaw:.3f}')
                        self.state = self.STATE_IDLE

        except Exception as exc:                                             # never leave the base coasting
            self.get_logger().error(f'process_navigation: {exc}')
            self.send(0.0, 0.0)

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_nav class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('ebot_nav_process')                    # creating ROS node

    node.get_logger().info('Node created: eBot navigation process') # logging information

    ebot_nav_class = ebot_nav()                                     # creating a new object for class 'ebot_nav'

    try:
        rclpy.spin(ebot_nav_class)                                  # spin the node to keep it alive in ROS 2 DDS
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        ebot_nav_class.send(0.0, 0.0)                               # leave the base stopped
    ebot_nav_class.destroy_node()                                   # destroy node after spin ends
    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()                                            # shutdown process


if __name__ == '__main__':

    main()
