#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1C of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          6494
# Author List:      Krishang Nigam, Masum Pancholi, Harshil Jayswal, Divy Vaghasiya
# Filename:         task1c.py
# Functions:        wrap_angle, clamp, yaw_from_quat, ebot_nav.__init__, ebot_nav.pathcb,
#                   ebot_nav.odomcb, ebot_nav.scancb, ebot_nav.mapcb, ebot_nav.sector_min,
#                   ebot_nav.map_bias, ebot_nav.obstacle_avoidance, ebot_nav.publish_command,
#                   ebot_nav.stop_robot, ebot_nav.follow_route, ebot_nav.process_navigation,
#                   main
# Nodes:            ebot_nav_node
#
# Publishing Topics  - [ /cmd_vel ]
# Subscribing Topics - [ /ebot_path, /odom, /scan, /map ]


################### IMPORT MODULES #######################

import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


##################### TASK CONSTANTS #######################

path_topic = '/ebot_path'
odom_topic = '/odom'
scan_topic = '/scan'
map_topic = '/map'
cmd_topic = '/cmd_vel'

# What /cmd_vel accepts.
max_linear_mps = 0.5
max_angular_rps = 1.0


##################### CONTROLLER TUNING #######################

kp_heading = 1.8                        # heading proportional gain
cruise_speed = 0.40                     # m/s when facing the waypoint
waypoint_tolerance = 0.25               # m (scoring limit 0.5, bonus 0.3)
final_tolerance = 0.18                  # m, tighter for the last waypoint
rotate_only_angle = math.radians(45.0)  # turn on the spot above this heading error
slow_radius = 0.80                      # m: slow down inside this distance of a waypoint
min_slow_factor = 0.25

# Command smoothing: largest change per 50 ms cycle.
max_linear_change = 0.05
max_angular_change = 0.12


##################### OBSTACLE TUNING #######################

obstacle_slow_distance = 0.90           # m: start slowing and steering
obstacle_stop_distance = 0.38           # m: no forward motion inside this
front_half_angle = math.radians(20.0)
side_angles = (math.radians(20.0), math.radians(65.0))
avoid_turn_gain = 0.65                  # rad/s at full proximity
min_avoid_turn = 0.35                   # rad/s while stopped in front of a rock
max_avoid_bias = 0.8

map_check_radius = 1.0                  # m around the eBot considered on /map
map_corridor_half_width = 0.85          # m either side of the heading
map_occupied_threshold = 65
map_bias_gain = 0.18


##################### HELPER FUNCTIONS #######################

def wrap_angle(angle):
    '''
    Description:    Wraps an angle to [-pi, pi].
    '''
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, low, high):
    '''
    Description:    Limits a value to [low, high].
    '''
    return max(low, min(high, value))


def yaw_from_quat(q):
    '''
    Description:    Yaw of a geometry_msgs Quaternion.
    '''
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Drives the eBot along the route on /ebot_path, through its waypoints
                    in order, steering round rocks seen on /scan and /map.
    '''

    # State machine definitions
    STATE_WAIT_FOR_DATA = 0
    STATE_FOLLOW = 1
    STATE_COMPLETED = 2

    def __init__(self):
        '''
        Description:    Initialization of class ebot_nav
        '''
        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # /ebot_path and /map are latched: published once, held for late subscribers.
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############
        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scancb, qos_profile_sensor_data)
        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self.mapcb, latched)

        ############ Topic PUBLISHERS ############
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        ############ State Variables ############
        self.route = None                       # [(x, y), ...]
        self.odom = None                        # (x, y, yaw)
        self.scan = None
        self.scan_angles = None                 # beam angles, cached per scan geometry
        self.rocks_xy = None                    # (N, 2) world coordinates of occupied cells

        self.state = self.STATE_WAIT_FOR_DATA
        self.current_waypoint = 0
        self.previous_linear = 0.0
        self.previous_angular = 0.0
        self.obstacle_active = False

        ############ Control Timer ############
        self.control_period = 0.05               # 20 Hz
        self.timer = self.create_timer(self.control_period, self.process_navigation)
        self.get_logger().info('eBot navigator running on simulation clock.')

    # ------------------------------------------------------------------ callbacks

    def pathcb(self, data):
        '''
        Description:    Stores the route once; a replacement mid-run is ignored.

        Args:
            data (Path):    The route, as a sequence of poses
        '''
        if self.route is not None or not data.poses:
            return
        self.route = [(p.pose.position.x, p.pose.position.y) for p in data.poses]
        self.get_logger().info(f'Route: {len(self.route)} waypoints in frame {data.header.frame_id}')

    def odomcb(self, data):
        '''
        Description:    Stores the eBot's position and heading.

        Args:
            data (Odometry):    Pose and velocity of the base
        '''
        p = data.pose.pose.position
        self.odom = (p.x, p.y, yaw_from_quat(data.pose.pose.orientation))

    def scancb(self, data):
        '''
        Description:    Stores the latest lidar sweep and caches its beam angles.

        Args:
            data (LaserScan):   One lidar sweep
        '''
        if self.scan_angles is None or len(self.scan_angles) != len(data.ranges):
            self.scan_angles = data.angle_min + np.arange(len(data.ranges)) * data.angle_increment
        self.scan = data

    def mapcb(self, data):
        '''
        Description:    Converts the occupancy grid, once, into world coordinates of its
                        occupied cells, so the per-cycle check is a vector operation.

        Args:
            data (OccupancyGrid):   The rock layout
        '''
        info = data.info
        grid = np.asarray(data.data, dtype=np.int16).reshape(info.height, info.width)
        rows, cols = np.nonzero(grid >= map_occupied_threshold)
        local = np.stack([(cols + 0.5) * info.resolution, (rows + 0.5) * info.resolution], axis=1)
        o = info.origin
        yaw = yaw_from_quat(o.orientation)
        c, s = math.cos(yaw), math.sin(yaw)
        self.rocks_xy = local @ np.array([[c, s], [-s, c]]) + (o.position.x, o.position.y)
        self.get_logger().info(f'Map: {info.width}x{info.height} at {info.resolution} m, '
                               f'{len(self.rocks_xy)} occupied cells')

    # ------------------------------------------------------------------ sensing

    def sector_min(self, low, high):
        '''
        Description:    Nearest valid lidar return between two beam angles.

        Args:
            low, high (float):  Sector bounds in radians (0 is straight ahead)

        Returns:
            float or None:      Range in metres, or None when the sector is empty
        '''
        r = np.asarray(self.scan.ranges, dtype=np.float64)
        ok = ((self.scan_angles >= low) & (self.scan_angles <= high) & np.isfinite(r) &
              (r >= self.scan.range_min) & (r <= self.scan.range_max))
        return float(r[ok].min()) if ok.any() else None

    def map_bias(self):
        '''
        Description:    Which side the mapped rocks ahead crowd: weighted count of occupied
                        cells left minus right, within map_check_radius.

        Returns:
            float:  Positive prefers turning left, negative right, in [-1, 1]
        '''
        if self.rocks_xy is None or len(self.rocks_xy) == 0:
            return 0.0
        x, y, yaw = self.odom
        d = self.rocks_xy - (x, y)
        dist = np.hypot(d[:, 0], d[:, 1])
        forward = math.cos(yaw) * d[:, 0] + math.sin(yaw) * d[:, 1]
        left = -math.sin(yaw) * d[:, 0] + math.cos(yaw) * d[:, 1]
        keep = (dist > 0.05) & (dist <= map_check_radius) & (forward > 0.0) & \
               (np.abs(left) <= map_corridor_half_width)
        weight = (1.0 - dist[keep] / map_check_radius) ** 2
        bias = weight[left[keep] > 0.0].sum() - weight[left[keep] <= 0.0].sum()
        return clamp(float(bias), -1.0, 1.0)

    def obstacle_avoidance(self, front):
        '''
        Description:    Speed scale and steering bias from the lidar and the map.

        Args:
            front (float or None):  Nearest return in the front sector

        Returns:
            tuple:  (linear_scale, angular_bias, obstacle_detected)
        '''
        linear_scale, angular_bias, detected = 1.0, 0.0, False
        if front is not None and front < obstacle_slow_distance:
            detected = True
            span = obstacle_slow_distance - obstacle_stop_distance
            linear_scale = clamp((front - obstacle_stop_distance) / span, 0.0, 1.0)
            left = self.sector_min(side_angles[0], side_angles[1])
            right = self.sector_min(-side_angles[1], -side_angles[0])
            left = 8.0 if left is None else left            # no return = more room
            right = 8.0 if right is None else right
            direction = 1.0 if left > right else -1.0
            proximity = clamp((obstacle_slow_distance - front) / span, 0.0, 1.0)
            angular_bias = direction * avoid_turn_gain * proximity
        angular_bias += map_bias_gain * self.map_bias()
        return linear_scale, clamp(angular_bias, -max_avoid_bias, max_avoid_bias), detected

    # ------------------------------------------------------------------ commands

    def publish_command(self, linear, angular):
        '''
        Description:    Publishes a Twist, limited to the base's range and smoothed so it
                        changes by at most max_*_change per cycle.
        '''
        linear = clamp(linear, -max_linear_mps, max_linear_mps)
        angular = clamp(angular, -max_angular_rps, max_angular_rps)
        linear = self.previous_linear + clamp(linear - self.previous_linear, -max_linear_change, max_linear_change)
        angular = self.previous_angular + clamp(angular - self.previous_angular, -max_angular_change, max_angular_change)
        self.previous_linear, self.previous_angular = linear, angular
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        '''
        Description:    Publishes zero velocity and resets the smoothing.
        '''
        self.previous_linear = self.previous_angular = 0.0
        self.cmd_pub.publish(Twist())

    # ------------------------------------------------------------------ control

    def follow_route(self):
        '''
        Description:    One control cycle towards the current waypoint.
        '''
        x, y, yaw = self.odom
        tx, ty = self.route[self.current_waypoint]
        distance = math.hypot(tx - x, ty - y)
        last = self.current_waypoint == len(self.route) - 1

        # Waypoint check
        if last and distance <= final_tolerance:
            self.stop_robot()
            self.state = self.STATE_COMPLETED
            self.get_logger().info(f'*** FINAL WAYPOINT REACHED ({distance:.3f} m). ROUTE COMPLETE ***')
            return
        if not last and distance <= waypoint_tolerance:
            self.current_waypoint += 1
            self.previous_linear = self.previous_angular = 0.0   # next segment starts cleanly
            self.get_logger().info(f'Waypoint {self.current_waypoint}/{len(self.route)} reached '
                                   f'({distance:.3f} m)')
            return

        # Heading control
        heading_error = wrap_angle(math.atan2(ty - y, tx - x) - yaw)
        angular = clamp(kp_heading * heading_error, -max_angular_rps, max_angular_rps)
        linear = 0.0 if abs(heading_error) > rotate_only_angle else cruise_speed * max(0.0, math.cos(heading_error))
        if distance < slow_radius:
            linear *= clamp(distance / slow_radius, min_slow_factor, 1.0)

        # Obstacle control
        front = self.sector_min(-front_half_angle, front_half_angle)
        scale, turn, detected = self.obstacle_avoidance(front)
        linear *= scale
        angular = clamp(angular + turn, -max_angular_rps, max_angular_rps)
        if front is not None and front <= obstacle_stop_distance:
            linear = 0.0                                     # never drive into a close rock
            angular = math.copysign(max(abs(angular), min_avoid_turn), angular if angular else 1.0)

        if detected and not self.obstacle_active:
            self.get_logger().info('Obstacle ahead: avoidance active')
        self.obstacle_active = detected
        self.publish_command(linear, angular)

    def process_navigation(self):
        '''
        Description:    Timer function: drive the route, stopping safely on any error.
        '''
        try:
            if self.state == self.STATE_WAIT_FOR_DATA:
                if not (self.route and self.odom is not None and self.scan is not None):
                    self.stop_robot()
                    return
                self.state = self.STATE_FOLLOW
                self.get_logger().info(f'Following route; target waypoint 1/{len(self.route)}')
            if self.state == self.STATE_FOLLOW:
                self.follow_route()
            else:
                self.stop_robot()
        except Exception as exc:                            # one bad cycle must not leave the base coasting
            self.get_logger().error(f'process_navigation: {exc}')
            self.stop_robot()


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_nav class to perform its task
    '''
    rclpy.init(args=sys.argv)
    node = ebot_nav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
