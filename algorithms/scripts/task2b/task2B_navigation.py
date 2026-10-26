#!/usr/bin/env python3


'''
*****************************************************************************************
*
*                       ===============================================
*                           StrataCobot (SC) Theme (eYRC 2026-27)
*                       ===============================================
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
# Author List:      Krishang Nigam, Masum Pancholi, Harshil Jayswal, Divy Vaghasiya
# Filename:         task2B_navigation.py
# Functions:        pathcb, odomcb, wrap_angle, clamp, smooth_command,
#                   publish_command, stop_robot, update_target,
#                   process_navigation
# Nodes:            ebot_nav_node
#
# Publishing Topics  - [ /cmd_vel ]
# Subscribing Topics - [ /ebot_path, /odom ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math

from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy
)


##################### TASK CONSTANTS #######################

path_topic = '/ebot_path'
odom_topic = '/odom'
cmd_topic = '/cmd_vel'

# Velocity limits specified by the task interface.
max_linear_mps = 0.5
max_angular_rps = 1.0

# Home position in the map frame. At home only the pose's own yaw is valid;
# at the ore drop pose and the arm pose either yaw (0 or pi) is valid.
home_xy = (0.0, 0.0)
HOME_RADIUS = 0.5

# At home the follower aims this far away from the left wall (+x), so the turn in
# place has more room. The published path itself still ends exactly at home.
HOME_AIM_SHIFT = 0.06


# ---------------- Controller parameters ---------------- #

# Heading proportional gain.
KP_HEADING = 1.8

# Cruise speed along the path.
CRUISE_SPEED = 0.40

# Look-ahead distance used to pick the point the eBot steers towards.
LOOKAHEAD = 0.35

# How many waypoints ahead of the current progress are searched
# when finding the nearest point on the path.
SEARCH_WINDOW = 40

# The eBot is "at the end" when it is this close to the last waypoint.
FINAL_TOLERANCE = 0.10

# Slow down only inside this distance of the last waypoint.
FINAL_SLOW_DISTANCE = 0.80

# If the target direction is far away from the current heading,
# rotate first rather than driving strongly forward.
ROTATE_ONLY_ANGLE = math.radians(45.0)

# At the start of every route the eBot first turns in place until it points along the
# path (within this error), and only then drives. This stops it from arcing sideways
# into the wall at the arm pose and at home.
START_TURN_TOLERANCE = math.radians(10.0)

# Final yaw alignment: stop when within this error (limit is 0.15 rad).
YAW_TOLERANCE = 0.05
KP_ALIGN = 1.5
MAX_ALIGN_RATE = 0.6


# ---------------- Command smoothing ---------------- #

MAX_LINEAR_CHANGE = 0.05
MAX_ANGULAR_CHANGE = 0.12


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along each route
                    published on /ebot_path, stopping and waiting after every route.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_nav
        '''

        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.Parameter.Type.BOOL,
                    True
                )
            ]
        )


        ############ Topic SUBSCRIPTIONS ############

        route_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.path_sub = self.create_subscription(
            Path,
            path_topic,
            self.pathcb,
            route_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odomcb,
            10
        )


        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(
            Twist,
            cmd_topic,
            10
        )


        ############ Constructor VARIABLES/OBJECTS ############

        # 20 Hz control loop.
        control_rate = 0.05
        self.timer = self.create_timer(
            control_rate,
            self.process_navigation
        )

        # Received route: list of (x, y).
        self.route = None

        # Current odometry state: (x, y, yaw).
        self.odom = None


        ############ ADD YOUR CODE HERE ############

        # Index of the nearest route point reached so far (never decreases).
        self.progress = 0

        # Frame reported by /ebot_path.
        self.route_frame = None

        # Yaw of the last pose of the route, and whether the opposite yaw
        # (that value plus pi) is also valid at the end of this route.
        self.final_yaw = 0.0
        self.either_yaw = True

        # True while turning in place to face along the start of a new route.
        self.start_turning = True

        # True while turning in place to the final yaw.
        self.aligning = False

        # True once the current route is complete; the eBot then waits
        # for the next route.
        self.finished = False

        # Previous command values used to avoid abrupt changes.
        self.previous_linear = 0.0
        self.previous_angular = 0.0

        ############################################


    ##################### ANGLE / VALUE HELPERS #######################

    def wrap_angle(self, angle):
        '''
        Description:    Wrap an angle to the range [-pi, pi].

        Args:
            angle (float):    Angle in radians

        Returns:
            float:            Wrapped angle
        '''

        return math.atan2(math.sin(angle), math.cos(angle))


    def clamp(self, value, low, high):
        '''
        Description:    Limit a value to a specified range.

        Args:
            value (float):    Value to limit
            low (float):      Minimum value
            high (float):     Maximum value

        Returns:
            float:            Limited value
        '''

        return max(low, min(high, value))


    ##################### COMMAND HANDLING #######################

    def smooth_command(self, desired_linear, desired_angular):
        '''
        Description:    Limit command changes between control cycles.

        Args:
            desired_linear (float):     Requested linear speed
            desired_angular (float):    Requested angular speed

        Returns:
            tuple:                      Rate-limited (linear, angular)
        '''

        linear = self.previous_linear + self.clamp(
            desired_linear - self.previous_linear,
            -MAX_LINEAR_CHANGE,
            MAX_LINEAR_CHANGE
        )

        angular = self.previous_angular + self.clamp(
            desired_angular - self.previous_angular,
            -MAX_ANGULAR_CHANGE,
            MAX_ANGULAR_CHANGE
        )

        linear = self.clamp(linear, -max_linear_mps, max_linear_mps)
        angular = self.clamp(angular, -max_angular_rps, max_angular_rps)

        self.previous_linear = linear
        self.previous_angular = angular

        return linear, angular


    def publish_command(self, linear_x, angular_z):
        '''
        Description:    Publish a rate-limited velocity command to /cmd_vel.

        Args:
            linear_x (float):     Requested linear speed
            angular_z (float):    Requested angular speed
        '''

        linear_x = self.clamp(linear_x, -max_linear_mps, max_linear_mps)
        angular_z = self.clamp(angular_z, -max_angular_rps, max_angular_rps)

        linear_x, angular_z = self.smooth_command(linear_x, angular_z)

        command = Twist()
        command.linear.x = linear_x
        command.angular.z = angular_z

        self.cmd_pub.publish(command)


    def stop_robot(self):
        '''
        Description:    Stop the eBot by explicitly publishing zero velocity.
        '''

        self.previous_linear = 0.0
        self.previous_angular = 0.0

        command = Twist()
        command.linear.x = 0.0
        command.angular.z = 0.0

        self.cmd_pub.publish(command)


    ##################### PATH CALLBACK #######################

    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
                        Every new route replaces the old one and restarts following.

        Args:
            data (Path):    The route, as a sequence of poses

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            if len(data.poses) == 0:

                self.get_logger().warning('Received empty /ebot_path')

                return

            route = []

            for waypoint in data.poses:

                position = waypoint.pose.position

                route.append((float(position.x), float(position.y)))

            # Final yaw comes from the last pose of the route.
            last_orientation = data.poses[-1].pose.orientation

            self.final_yaw = math.atan2(
                2.0 * (last_orientation.w * last_orientation.z
                       + last_orientation.x * last_orientation.y),
                1.0 - 2.0 * (last_orientation.y * last_orientation.y
                             + last_orientation.z * last_orientation.z)
            )

            # Home accepts only its own yaw; other end poses accept it or its opposite.
            end_x, end_y = route[-1]

            self.either_yaw = (
                math.hypot(end_x - home_xy[0], end_y - home_xy[1]) > HOME_RADIUS
            )

            # The follower's own copy of the route stops slightly off the wall at home.
            if not self.either_yaw:
                route[-1] = (end_x + HOME_AIM_SHIFT, end_y)

            self.route = route
            self.route_frame = data.header.frame_id

            self.progress = 0
            self.start_turning = True
            self.aligning = False
            self.finished = False

            self.previous_linear = 0.0
            self.previous_angular = 0.0

            self.get_logger().info(
                f'Received route: {len(self.route)} waypoints '
                f'in frame {self.route_frame}, final yaw {self.final_yaw:.2f} rad, '
                f'either yaw valid: {self.either_yaw}'
            )

        except Exception as error:

            self.get_logger().error(f'Path callback error: {error}')

            self.route = None

        ############################################


    ##################### ODOM CALLBACK #######################

    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.

        Args:
            data (Odometry):    Pose and velocity of the base

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            position = data.pose.pose.position
            orientation = data.pose.pose.orientation

            qx = float(orientation.x)
            qy = float(orientation.y)
            qz = float(orientation.z)
            qw = float(orientation.w)

            # Quaternion -> yaw.
            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz)
            )

            self.odom = (float(position.x), float(position.y), yaw)

        except Exception as error:

            self.get_logger().error(f'Odometry callback error: {error}')

        ############################################


    ##################### TARGET SELECTION #######################

    def update_target(self, x, y):
        '''
        Description:    Advance progress to the nearest route point, then pick
                        a look-ahead point to steer towards.

        Args:
            x (float):    Current x position
            y (float):    Current y position

        Returns:
            tuple:        (target_x, target_y)
        '''

        last = len(self.route) - 1

        window_end = min(last, self.progress + SEARCH_WINDOW)

        best_index = self.progress
        best_distance = float('inf')

        for index in range(self.progress, window_end + 1):

            px, py = self.route[index]

            distance = math.hypot(px - x, py - y)

            if distance < best_distance:

                best_distance = distance
                best_index = index

        self.progress = best_index

        target_index = self.progress

        while target_index < last:

            px, py = self.route[target_index]

            if math.hypot(px - x, py - y) >= LOOKAHEAD:
                break

            target_index += 1

        return self.route[target_index]


    ##################### NAVIGATION #######################

    def process_navigation(self):
        '''
        Description:    Timer function used to drive the eBot along the route,
                        turn to the final yaw, and then wait for the next route.

        Args:

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            # Wait for a route and for odometry.
            if self.route is None or self.odom is None:
                self.stop_robot()
                return

            # Route complete: stay still until a new route arrives.
            if self.finished:
                self.stop_robot()
                return

            x, y, yaw = self.odom

            final_x, final_y = self.route[-1]

            final_distance = math.hypot(final_x - x, final_y - y)


            ################ FINAL ALIGNMENT ################

            if not self.aligning and final_distance <= FINAL_TOLERANCE:

                self.aligning = True

                self.get_logger().info(
                    f'End position reached, distance={final_distance:.3f} m'
                )

            if self.aligning:

                if self.either_yaw:

                    # Ore drop pose or arm pose: the route's final yaw or its opposite,
                    # whichever is nearer.
                    if abs(self.wrap_angle(yaw - self.final_yaw)) < math.pi / 2.0:
                        goal_yaw = self.final_yaw
                    else:
                        goal_yaw = self.wrap_angle(self.final_yaw + math.pi)

                else:

                    # Home: only the route's own final yaw is valid.
                    goal_yaw = self.final_yaw

                yaw_error = self.wrap_angle(goal_yaw - yaw)

                if abs(yaw_error) < YAW_TOLERANCE:

                    self.aligning = False
                    self.finished = True

                    self.stop_robot()

                    self.get_logger().info(
                        f'Aligned and stopped at ({x:.3f}, {y:.3f}), yaw {yaw:.3f} rad.'
                    )

                    return

                self.publish_command(
                    0.0,
                    self.clamp(
                        KP_ALIGN * yaw_error,
                        -MAX_ALIGN_RATE,
                        MAX_ALIGN_RATE
                    )
                )

                return


            ################ TARGET HEADING ################

            target_x, target_y = self.update_target(x, y)

            target_heading = math.atan2(target_y - y, target_x - x)

            heading_error = self.wrap_angle(target_heading - yaw)


            ################ TURN TO FACE THE PATH FIRST ################

            if self.start_turning:

                if abs(heading_error) > START_TURN_TOLERANCE:

                    self.publish_command(
                        0.0,
                        self.clamp(
                            KP_ALIGN * heading_error,
                            -MAX_ALIGN_RATE,
                            MAX_ALIGN_RATE
                        )
                    )

                    return

                self.start_turning = False


            ################ PATH CONTROL ################

            angular_cmd = self.clamp(
                KP_HEADING * heading_error,
                -max_angular_rps,
                max_angular_rps
            )

            if abs(heading_error) > ROTATE_ONLY_ANGLE:

                linear_cmd = 0.0

            else:

                linear_cmd = CRUISE_SPEED * max(0.0, math.cos(heading_error))

            # Slow down only when approaching the end of the route.
            if final_distance < FINAL_SLOW_DISTANCE:

                linear_cmd *= self.clamp(
                    final_distance / FINAL_SLOW_DISTANCE,
                    0.25,
                    1.0
                )


            ################ PUBLISH ################

            self.publish_command(linear_cmd, angular_cmd)

        except Exception as error:

            self.get_logger().error(f'Navigation error: {error}')

            self.stop_robot()

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_nav class to perform its task
    '''

    rclpy.init(args=sys.argv)

    ebot_nav_class = ebot_nav()

    ebot_nav_class.get_logger().info('Node created: eBot navigation process')

    try:

        rclpy.spin(ebot_nav_class)

    except KeyboardInterrupt:

        pass

    finally:

        if rclpy.ok():
            ebot_nav_class.stop_robot()

        ebot_nav_class.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':

    main()
