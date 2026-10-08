#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 2B of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task2B_navigation.py
# Functions:        wrap_angle, yaw_from_quat, clamp, ebot_nav.__init__, ebot_nav.pathcb,
#                   ebot_nav.odomcb, ebot_nav.scancb, ebot_nav.send, ebot_nav.turn_towards,
#                   ebot_nav.travel_heading, ebot_nav.align_target, ebot_nav.final_target,
#                   ebot_nav.seg_heading, ebot_nav.stop_after, ebot_nav.dist_to_next_stop,
#                   ebot_nav.carrot, ebot_nav.follow_segment, ebot_nav.process_navigation, main
# Nodes:            ebot_nav_node
#
# Publishing Topics  - [ /cmd_vel ]
# Subscribing Topics - [ /ebot_path, /odom, /scan ]


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


##################### TASK CONSTANTS #######################

# The planner publishes a path here for each step, held for late subscribers.
path_topic = '/ebot_path'

odom_topic = '/odom'
scan_topic = '/scan'
cmd_topic = '/cmd_vel'

# What /cmd_vel accepts. The base clips anything beyond these, and clipping one of the
# two changes their ratio - which is the arc the base actually drives.
max_linear_mps = 0.5
max_angular_rps = 1.0


##################### CONTROLLER TUNING #######################

cruise_speed = 0.35             # m/s on straights
min_speed = 0.06                # m/s floor while approaching a stop point
lookahead = 0.30                # m: carrot distance, never past the next stop point
k_heading = 2.0                 # angular gain while driving
k_turn = 1.8                    # angular gain while turning on the spot
min_turn_rate = 0.18            # rad/s, so slip cannot stall a small final correction
max_turn_rate = 0.8             # rad/s while turning on the spot
turn_in_place = 0.45            # rad: heading error above this stops forward motion
corner_angle = 0.25             # rad: a bend sharper than this is taken stopped
align_tol = 0.04                # rad: done aligning to a segment
final_yaw_tol = 0.03            # rad: final heading (the evaluator allows 0.15)
final_pos_tol = 0.04            # m: final position (the evaluator allows 0.3)
slow_radius = 0.6               # m: start slowing this far before a stop point
reverse_bias = 0.3              # rad: drive a segment backwards only if that saves more turning than this

# Turning on the spot shifts the eBot's centre by several centimetres (a skid steer does
# not pivot cleanly). While turning, it is driven back along its own axis towards the
# point it should be turning on.
k_hold = 1.5                    # 1/s: forward/back speed per metre of drift along the axis
max_hold_speed = 0.08           # m/s

# Home must be faced at yaw 0; the ore drop pose and the arm pose accept 0 or pi.
home_xy = (0.0, 0.0)


##################### MATH UTILITIES #######################

def wrap_angle(a):
    '''
    Description:    Wraps an angle into [-pi, pi).
    '''
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q):
    '''
    Description:    Yaw of a geometry_msgs Quaternion.
    '''
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def clamp(value, low, high):
    '''
    Description:    Limits a value to [low, high].
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
    STATE_FINAL_TURN = 3        # on the last pose: turn to its yaw

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

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)            # the eBot drives on what is published here

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
        self.reverse = False            # driving the current segment backwards
        self.stop_xy = None             # where the eBot stopped at the end of the path
        self.state = self.STATE_IDLE

        ############################################


    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
                        Use this function to receive the waypoints the eBot has to drive.

        Args:
            data (Path):    The route, as a sequence of poses

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # A new path arrives for every step; each one is taken once.
        key = (data.header.stamp.sec, data.header.stamp.nanosec, len(data.poses))
        if len(data.poses) < 2 or key == self.last_key:
            return
        self.last_key = key
        route = [(data.poses[0].pose.position.x, data.poses[0].pose.position.y)]
        for p in data.poses[1:]:
            pt = (p.pose.position.x, p.pose.position.y)
            if math.hypot(pt[0] - route[-1][0], pt[1] - route[-1][1]) > 1e-4:   # drop zero-length segments
                route.append(pt)
        if len(route) < 2:
            return
        self.route = route
        self.final_yaw = yaw_from_quat(data.poses[-1].pose.orientation)
        self.seg = 0
        self.reverse = False
        self.state = self.STATE_ALIGN
        self.get_logger().info(f'New path: {len(route)} poses in frame {data.header.frame_id}, ends at '
                               f'({route[-1][0]:.2f}, {route[-1][1]:.2f}) yaw {self.final_yaw:.2f}')

        ############################################


    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Use this function to receive where the base currently is.

        Args:
            data (Odometry):    Pose and velocity of the base

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.yaw = yaw_from_quat(data.pose.pose.orientation)
        self.odom = data

        ############################################


    def scancb(self, data):
        '''
        Description:    Callback function for the lidar topic.
                        Use this function to receive what the lidar currently sees.

        Args:
            data (LaserScan):    One lidar sweep

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # Stored for debugging only. In Task 2B the eBot must follow the path it published,
        # so it never steers off it for a rock seen here; the planned path clears the rocks.
        self.scan = data

        ############################################


    def send(self, v, w):
        '''
        Description:    Publishes a Twist, scaled down as a whole if over the limits so the
                        v/w ratio - the arc driven - is kept.
        '''
        scale = min(1.0, max_linear_mps / abs(v) if v else 1.0, max_angular_rps / abs(w) if w else 1.0)
        cmd = Twist()
        cmd.linear.x = float(v * scale)
        cmd.angular.z = float(w * scale)
        self.cmd_pub.publish(cmd)


    def turn_towards(self, target_yaw, tol, hold):
        '''
        Description:    Turns on the spot towards target_yaw, closing the loop on /odom, while
                        driving along its own axis to stay on the 'hold' point.

        Args:
            target_yaw (float):     heading to reach, in radians
            tol (float):            done within this many radians
            hold (tuple):           (x, y) the eBot should be turning on

        Returns:
            bool:   True once within 'tol' (and stopped)
        '''
        err = wrap_angle(target_yaw - self.yaw)
        if abs(err) < tol:
            self.send(0.0, 0.0)
            return True
        w = clamp(k_turn * err, -max_turn_rate, max_turn_rate)
        w = w if abs(w) >= min_turn_rate else math.copysign(min_turn_rate, err)
        along = (hold[0] - self.x) * math.cos(self.yaw) + (hold[1] - self.y) * math.sin(self.yaw)
        self.send(clamp(k_hold * along, -max_hold_speed, max_hold_speed), w)
        return False


    def travel_heading(self):
        '''
        Description:    The direction the eBot moves in: its heading, or the opposite when
                        driving the current segment backwards.
        '''
        return self.yaw + (math.pi if self.reverse else 0.0)


    def align_target(self):
        '''
        Description:    Chooses forwards or backwards for the current segment, whichever needs
                        less turning on the spot, and returns the heading to turn to.
        '''
        h = self.seg_heading(self.seg)
        if self.seg == len(self.route) - 2 and math.dist(self.route[-1], home_xy) < 0.3:
            # last segment into home: drive it so the eBot arrives already facing yaw 0
            self.reverse = abs(wrap_angle(h - self.final_yaw)) > math.pi / 2
        else:
            forward = abs(wrap_angle(h - self.yaw))
            backward = abs(wrap_angle(h + math.pi - self.yaw))
            self.reverse = backward + reverse_bias < forward
        return h + (math.pi if self.reverse else 0.0)


    def final_target(self):
        '''
        Description:    The yaw to finish on: exactly the path's yaw at home, otherwise the
                        nearer of the two allowed (it or its opposite).
        '''
        if math.dist(self.route[-1], home_xy) < 0.3:
            return self.final_yaw
        return min((self.final_yaw, self.final_yaw + math.pi), key=lambda y: abs(wrap_angle(y - self.yaw)))


    def seg_heading(self, i):
        '''
        Description:    Direction of segment i, in radians.
        '''
        (ax, ay), (bx, by) = self.route[i], self.route[i + 1]
        return math.atan2(by - ay, bx - ax)


    def stop_after(self, i):
        '''
        Description:    True when the eBot must stop at the end of segment i: the last
                        segment, or a bend sharper than corner_angle.
        '''
        return i >= len(self.route) - 2 or \
            abs(wrap_angle(self.seg_heading(i + 1) - self.seg_heading(i))) > corner_angle


    def dist_to_next_stop(self, i, along):
        '''
        Description:    Distance left along the route, from 'along' metres into segment i,
                        to the next stop point.
        '''
        d = math.dist(self.route[i], self.route[i + 1]) - along
        while not self.stop_after(i):
            i += 1
            d += math.dist(self.route[i], self.route[i + 1])
        return max(d, 0.0)


    def carrot(self, i, along, dist):
        '''
        Description:    The point 'dist' metres further along the route from 'along' metres
                        into segment i. It never turns past the next stop point, so bends are
                        never cut; near a stop point it runs on along the same straight line,
                        so the heading stays steady right up to the stop.
        '''
        (ax, ay), (bx, by) = self.route[i], self.route[i + 1]
        seg_len = math.hypot(bx - ax, by - ay)
        target = along + dist
        while target > seg_len and not self.stop_after(i):
            target -= seg_len
            i += 1
            (ax, ay), (bx, by) = self.route[i], self.route[i + 1]
            seg_len = math.hypot(bx - ax, by - ay)
        t = max(target / seg_len, 0.0)
        return (ax + (bx - ax) * t, ay + (by - ay) * t)


    def follow_segment(self):
        '''
        Description:    One control cycle along the current segment: pure pursuit on a carrot
                        that never passes the next stop point, slowing towards it.
        '''
        i = self.seg
        (ax, ay), (bx, by) = self.route[i], self.route[i + 1]
        seg_len = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / seg_len, (by - ay) / seg_len
        along = (self.x - ax) * ux + (self.y - ay) * uy
        last = i >= len(self.route) - 2
        to_end = math.hypot(bx - self.x, by - self.y)

        # End of the segment: finish, carry straight on, or stop and turn at a sharp bend.
        if along >= seg_len - 1e-3 or (last and to_end < final_pos_tol):
            if last:
                self.send(0.0, 0.0)
                self.stop_xy = (self.x, self.y)     # turn on the spot here, not on the exact pose
                self.state = self.STATE_FINAL_TURN
            else:
                sharp = self.stop_after(i)
                self.seg += 1
                if sharp:
                    self.send(0.0, 0.0)
                    self.state = self.STATE_ALIGN
            return

        cx, cy = self.carrot(i, max(along, 0.0), lookahead)
        err = wrap_angle(math.atan2(cy - self.y, cx - self.x) - self.travel_heading())

        if abs(err) > turn_in_place and to_end > final_pos_tol:              # badly off heading: turn first
            self.send(0.0, clamp(k_turn * err, -max_turn_rate, max_turn_rate))
            return

        v = cruise_speed * min(1.0, self.dist_to_next_stop(i, max(along, 0.0)) / slow_radius)
        v = max(v, min_speed) * max(0.0, math.cos(err))
        self.send(-v if self.reverse else v, clamp(k_heading * err, -max_angular_rps, max_angular_rps))


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
                if self.turn_towards(self.align_target(), align_tol, self.route[self.seg]):
                    self.state = self.STATE_FOLLOW

            # ------------------------------------------------------------------
            # STATE 2: Drive along the segments
            # ------------------------------------------------------------------
            elif self.state == self.STATE_FOLLOW:
                self.follow_segment()

            # ------------------------------------------------------------------
            # STATE 3: On the last pose, turn to its yaw, then hold still
            # ------------------------------------------------------------------
            elif self.state == self.STATE_FINAL_TURN:
                if self.turn_towards(self.final_target(), final_yaw_tol, self.stop_xy):
                    self.get_logger().info(f'Arrived at ({self.x:.3f}, {self.y:.3f}) yaw {self.yaw:.3f}')
                    self.state = self.STATE_IDLE

        except Exception as exc:                                             # one bad cycle must not leave the base coasting
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
        rclpy.spin(ebot_nav_class)                                  # spining on the object to make it alive in ROS 2 DDS
    except KeyboardInterrupt:
        pass

    ebot_nav_class.send(0.0, 0.0)                                   # leave the base stopped
    ebot_nav_class.destroy_node()                                   # destroy node after spin ends
    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()                                            # shutdown process


if __name__ == '__main__':

    main()
