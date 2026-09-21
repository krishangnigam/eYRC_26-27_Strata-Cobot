#!/usr/bin/env python3


'''
*****************************************************************************************
*
*                       ===============================================
*                           StrataCobot (SC) Theme (eYRC 2026-27)
*                       ===============================================
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
# Functions:        pathcb, odomcb, scancb, mapcb,
#                   wrap_angle, clamp, smooth_command,
#                   publish_command, stop_robot,
#                   get_scan_sector_min, get_lidar_obstacle_info,
#                   map_to_world, get_map_obstacle_bias,
#                   obstacle_avoidance, process_navigation
# Nodes:            ebot_nav_node
#
# Publishing Topics  - [ /cmd_vel ]
# Subscribing Topics - [ /ebot_path, /odom, /scan, /map ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math

from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data
)
from sensor_msgs.msg import LaserScan


##################### TASK CONSTANTS #######################

# The route is published once when the world loads and retained
# for late subscribers.
path_topic = '/ebot_path'

odom_topic = '/odom'
scan_topic = '/scan'
map_topic = '/map'
cmd_topic = '/cmd_vel'

# Velocity limits specified by the task interface.
max_linear_mps = 0.5
max_angular_rps = 1.0


# ---------------- Controller parameters ---------------- #

# Heading proportional gain.
KP_HEADING = 1.8

# Waypoint acceptance distances.
# The task scoring limit is 0.5 m and the bonus limit is 0.3 m.
WAYPOINT_TOLERANCE = 0.25
FINAL_TOLERANCE = 0.18

# If the target direction is far away from the current heading,
# rotate first rather than driving strongly forward.
ROTATE_ONLY_ANGLE = math.radians(45.0)


# ---------------- Obstacle parameters ---------------- #

OBSTACLE_SLOW_DISTANCE = 0.90
OBSTACLE_STOP_DISTANCE = 0.38

FRONT_HALF_ANGLE = math.radians(20.0)
SIDE_MIN_ANGLE = math.radians(20.0)
SIDE_MAX_ANGLE = math.radians(65.0)


# ---------------- Map parameters ---------------- #

MAP_CHECK_RADIUS = 1.0
MAP_OCCUPIED_THRESHOLD = 65


# ---------------- Command smoothing ---------------- #

MAX_LINEAR_CHANGE = 0.05
MAX_ANGULAR_CHANGE = 0.12


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along the route
                    published on /ebot_path, through its waypoints, in order.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_nav
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
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

        # /ebot_path is latched.
        # The publisher has already sent the route before this node starts,
        # so the subscription must request TRANSIENT_LOCAL durability.
        route_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # The route is received here.
        self.path_sub = self.create_subscription(
            Path,
            path_topic,
            self.pathcb,
            route_qos
        )

        # Current robot pose.
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odomcb,
            10
        )

        # Current lidar data.
        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scancb,
            qos_profile_sensor_data
        )

        # Occupancy map is also retained.
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self.mapcb,
            route_qos
        )


        ############ Topic PUBLISHERS ############

        # The eBot drives using linear.x and angular.z.
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

        # Received route.
        self.route = None

        # Current odometry state: (x, y, yaw).
        self.odom = None

        # Latest laser scan.
        self.scan = None

        # Occupancy grid.
        self.map_data = None


        ############ ADD YOUR CODE HERE ############

        # Index of the waypoint currently being followed.
        # The route is always followed in order.
        self.current_waypoint = 0

        # Frame reported by /ebot_path.
        self.route_frame = None

        # Becomes True after the last waypoint is reached.
        self.finished = False

        # Previous command values used to avoid abrupt changes.
        self.previous_linear = 0.0
        self.previous_angular = 0.0

        # Logging state.
        self.last_logged_waypoint = -1
        self.last_obstacle_state = False

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

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )


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

        return max(
            low,
            min(high, value)
        )


    ##################### COMMAND HANDLING #######################

    def smooth_command(self, desired_linear, desired_angular):
        '''
        Description:    Limit command changes between control cycles.
        '''

        linear_difference = (
            desired_linear
            - self.previous_linear
        )

        linear_difference = self.clamp(
            linear_difference,
            -MAX_LINEAR_CHANGE,
            MAX_LINEAR_CHANGE
        )

        angular_difference = (
            desired_angular
            - self.previous_angular
        )

        angular_difference = self.clamp(
            angular_difference,
            -MAX_ANGULAR_CHANGE,
            MAX_ANGULAR_CHANGE
        )

        linear = (
            self.previous_linear
            + linear_difference
        )

        angular = (
            self.previous_angular
            + angular_difference
        )

        linear = self.clamp(
            linear,
            -max_linear_mps,
            max_linear_mps
        )

        angular = self.clamp(
            angular,
            -max_angular_rps,
            max_angular_rps
        )

        self.previous_linear = linear
        self.previous_angular = angular

        return linear, angular


    def publish_command(self, linear_x, angular_z):
        '''
        Description:    Publish a velocity command to /cmd_vel.
        '''

        linear_x = self.clamp(
            linear_x,
            -max_linear_mps,
            max_linear_mps
        )

        angular_z = self.clamp(
            angular_z,
            -max_angular_rps,
            max_angular_rps
        )

        linear_x, angular_z = self.smooth_command(
            linear_x,
            angular_z
        )

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
                        Use this function to receive the waypoints the eBot has to drive.

        Args:
            data (Path):    The route, as a sequence of poses

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            # Keep the first valid route for this run.
            if self.route is not None:
                return

            if len(data.poses) == 0:

                self.get_logger().warning(
                    'Received empty /ebot_path'
                )

                return


            self.route = []

            for waypoint in data.poses:

                position = waypoint.pose.position

                self.route.append(
                    (
                        float(position.x),
                        float(position.y)
                    )
                )


            self.current_waypoint = 0
            self.route_frame = data.header.frame_id

            self.get_logger().info(
                f'Received route: '
                f'{len(self.route)} waypoints '
                f'in frame {self.route_frame}'
            )


        except Exception as error:

            self.get_logger().error(
                f'Path callback error: {error}'
            )

            self.route = None

        ############################################


    ##################### ODOM CALLBACK #######################

    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Use this function to receive where the base currently is.

        Args:
            data (Odometry):    Pose and velocity of the base

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            position = data.pose.pose.position
            orientation = data.pose.pose.orientation

            x = float(position.x)
            y = float(position.y)

            qx = float(orientation.x)
            qy = float(orientation.y)
            qz = float(orientation.z)
            qw = float(orientation.w)

            # Quaternion -> yaw.
            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz)
            )

            self.odom = (
                x,
                y,
                yaw
            )


        except Exception as error:

            self.get_logger().error(
                f'Odometry callback error: {error}'
            )

        ############################################


    ##################### LASER CALLBACK #######################

    def scancb(self, data):
        '''
        Description:    Callback function for the lidar topic.
                        Use this function to receive what the lidar currently sees.

        Args:
            data (LaserScan):    One lidar sweep

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            self.scan = data

        except Exception as error:

            self.get_logger().error(
                f'Laser callback error: {error}'
            )

        ############################################


    ##################### MAP CALLBACK #######################

    def mapcb(self, data):
        '''
        Description:    Callback function for the occupancy map.
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            self.map_data = data

        except Exception as error:

            self.get_logger().error(
                f'Map callback error: {error}'
            )

        ############################################


    ##################### LASER PROCESSING #######################

    def get_scan_sector_min(self, min_angle, max_angle):
        '''
        Description:    Find the nearest valid laser return in an angular sector.

        Args:
            min_angle (float):    Start angle in radians
            max_angle (float):    End angle in radians

        Returns:
            float or None:         Nearest valid range
        '''

        if self.scan is None:
            return None

        try:

            nearest = None

            for index, distance in enumerate(
                self.scan.ranges
            ):

                angle = (
                    self.scan.angle_min
                    + index * self.scan.angle_increment
                )

                if angle < min_angle or angle > max_angle:
                    continue

                if not math.isfinite(distance):
                    continue

                if distance < self.scan.range_min:
                    continue

                if distance > self.scan.range_max:
                    continue

                if nearest is None or distance < nearest:
                    nearest = distance

            return nearest

        except Exception:

            return None


    def get_lidar_obstacle_info(self):
        '''
        Description:    Read front, left and right lidar clearance.
        '''

        front = self.get_scan_sector_min(
            -FRONT_HALF_ANGLE,
            FRONT_HALF_ANGLE
        )

        left = self.get_scan_sector_min(
            SIDE_MIN_ANGLE,
            SIDE_MAX_ANGLE
        )

        right = self.get_scan_sector_min(
            -SIDE_MAX_ANGLE,
            -SIDE_MIN_ANGLE
        )

        return front, left, right


    ##################### MAP PROCESSING #######################

    def map_to_world(self, cell_x, cell_y):
        '''
        Description:    Convert occupancy-grid cell coordinates into world coordinates.
        '''

        if self.map_data is None:
            return None

        try:

            information = self.map_data.info

            resolution = float(
                information.resolution
            )

            if resolution <= 0.0:
                return None

            origin_x = float(
                information.origin.position.x
            )

            origin_y = float(
                information.origin.position.y
            )

            orientation = information.origin.orientation

            origin_yaw = math.atan2(
                2.0 * (
                    orientation.w * orientation.z
                    + orientation.x * orientation.y
                ),
                1.0 - 2.0 * (
                    orientation.y * orientation.y
                    + orientation.z * orientation.z
                )
            )

            local_x = (
                (cell_x + 0.5)
                * resolution
            )

            local_y = (
                (cell_y + 0.5)
                * resolution
            )

            world_x = (
                origin_x
                + math.cos(origin_yaw) * local_x
                - math.sin(origin_yaw) * local_y
            )

            world_y = (
                origin_y
                + math.sin(origin_yaw) * local_x
                + math.cos(origin_yaw) * local_y
            )

            return world_x, world_y

        except Exception:

            return None


    def get_map_obstacle_bias(self):
        '''
        Description:
            Estimate whether occupied map cells ahead of the robot
            are concentrated more on the left or right.

        Returns:
            float:
                negative -> prefer right
                positive -> prefer left
                zero     -> no useful information
        '''

        if self.map_data is None or self.odom is None:
            return 0.0

        try:

            x, y, yaw = self.odom

            info = self.map_data.info

            width = int(info.width)
            height = int(info.height)
            resolution = float(info.resolution)

            if (
                width <= 0
                or height <= 0
                or resolution <= 0.0
            ):
                return 0.0


            origin_orientation = info.origin.orientation

            origin_yaw = math.atan2(
                2.0 * (
                    origin_orientation.w
                    * origin_orientation.z
                    + origin_orientation.x
                    * origin_orientation.y
                ),
                1.0 - 2.0 * (
                    origin_orientation.y
                    * origin_orientation.y
                    + origin_orientation.z
                    * origin_orientation.z
                )
            )


            # Convert robot world position into map-local coordinates.
            dx_world = (
                x
                - info.origin.position.x
            )

            dy_world = (
                y
                - info.origin.position.y
            )

            local_x = (
                math.cos(origin_yaw) * dx_world
                + math.sin(origin_yaw) * dy_world
            )

            local_y = (
                -math.sin(origin_yaw) * dx_world
                + math.cos(origin_yaw) * dy_world
            )


            robot_cell_x = int(
                math.floor(
                    local_x / resolution
                )
            )

            robot_cell_y = int(
                math.floor(
                    local_y / resolution
                )
            )


            search_radius = int(
                math.ceil(
                    MAP_CHECK_RADIUS
                    / resolution
                )
            )

            left_weight = 0.0
            right_weight = 0.0

            minimum_x = max(
                0,
                robot_cell_x - search_radius
            )

            maximum_x = min(
                width - 1,
                robot_cell_x + search_radius
            )

            minimum_y = max(
                0,
                robot_cell_y - search_radius
            )

            maximum_y = min(
                height - 1,
                robot_cell_y + search_radius
            )


            for cell_y in range(
                minimum_y,
                maximum_y + 1
            ):

                row_start = (
                    cell_y * width
                )

                for cell_x in range(
                    minimum_x,
                    maximum_x + 1
                ):

                    index = (
                        row_start
                        + cell_x
                    )

                    if (
                        index < 0
                        or index >= len(
                            self.map_data.data
                        )
                    ):
                        continue

                    occupancy = (
                        self.map_data.data[index]
                    )

                    if occupancy < MAP_OCCUPIED_THRESHOLD:
                        continue

                    world_position = self.map_to_world(
                        cell_x,
                        cell_y
                    )

                    if world_position is None:
                        continue

                    obstacle_x, obstacle_y = (
                        world_position
                    )

                    dx = obstacle_x - x
                    dy = obstacle_y - y

                    distance = math.hypot(
                        dx,
                        dy
                    )

                    if distance <= 0.05:
                        continue

                    if distance > MAP_CHECK_RADIUS:
                        continue


                    # World coordinates -> robot coordinates.
                    forward = (
                        math.cos(yaw) * dx
                        + math.sin(yaw) * dy
                    )

                    left = (
                        -math.sin(yaw) * dx
                        + math.cos(yaw) * dy
                    )


                    # Only use cells in front of the rover.
                    if forward <= 0.0:
                        continue

                    if abs(left) > 0.85:
                        continue


                    weight = (
                        1.0
                        - distance / MAP_CHECK_RADIUS
                    )

                    weight *= weight


                    if left > 0.0:
                        left_weight += weight
                    else:
                        right_weight += weight


            bias = (
                left_weight
                - right_weight
            )

            return self.clamp(
                bias,
                -1.0,
                1.0
            )

        except Exception:

            return 0.0


    ##################### OBSTACLE HANDLING #######################

    def obstacle_avoidance(self):
        '''
        Description:
            Calculate obstacle-related steering and speed reduction.

        Returns:
            tuple:
                linear_scale,
                angular_bias,
                obstacle_detected
        '''

        front, left, right = (
            self.get_lidar_obstacle_info()
        )

        map_bias = (
            self.get_map_obstacle_bias()
        )

        linear_scale = 1.0
        angular_bias = 0.0
        obstacle_detected = False


        if front is not None:

            if front < OBSTACLE_SLOW_DISTANCE:

                obstacle_detected = True

                # Reduce forward velocity as the obstacle gets nearer.
                linear_scale = self.clamp(
                    (
                        front
                        - OBSTACLE_STOP_DISTANCE
                    )
                    /
                    (
                        OBSTACLE_SLOW_DISTANCE
                        - OBSTACLE_STOP_DISTANCE
                    ),
                    0.0,
                    1.0
                )


                # If one side has no valid reading,
                # treat it as having more available room.
                if left is None:
                    left_clearance = 8.0
                else:
                    left_clearance = left

                if right is None:
                    right_clearance = 8.0
                else:
                    right_clearance = right


                if left_clearance > right_clearance:
                    turn_direction = 1.0
                else:
                    turn_direction = -1.0


                # Stronger steering as the obstacle becomes closer.
                proximity = self.clamp(
                    (
                        OBSTACLE_SLOW_DISTANCE
                        - front
                    )
                    /
                    (
                        OBSTACLE_SLOW_DISTANCE
                        - OBSTACLE_STOP_DISTANCE
                    ),
                    0.0,
                    1.0
                )


                angular_bias = (
                    turn_direction
                    * 0.65
                    * proximity
                )


        # Add a smaller map-based preference.
        angular_bias += (
            0.18 * map_bias
        )

        angular_bias = self.clamp(
            angular_bias,
            -0.8,
            0.8
        )

        return (
            linear_scale,
            angular_bias,
            obstacle_detected
        )


    ##################### NAVIGATION #######################

    def process_navigation(self):
        '''
        Description:    Timer function used to drive the eBot along the route.

        Args:

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            # Wait until every required feedback source has arrived.
            if self.route is None:
                self.stop_robot()
                return

            if self.odom is None:
                self.stop_robot()
                return

            if self.scan is None:
                self.stop_robot()
                return

            if len(self.route) == 0:
                self.stop_robot()
                return


            # The route is already complete.
            if self.finished:
                self.stop_robot()
                return


            # Protect against an invalid waypoint index.
            if (
                self.current_waypoint
                >= len(self.route)
            ):

                self.finished = True

                self.stop_robot()

                self.get_logger().info(
                    'All waypoints completed. '
                    'Robot stopped.'
                )

                return


            # Current rover state.
            x, y, yaw = self.odom

            # Current target waypoint.
            target_x, target_y = (
                self.route[
                    self.current_waypoint
                ]
            )


            error_x = target_x - x
            error_y = target_y - y

            distance = math.hypot(
                error_x,
                error_y
            )


            ################ WAYPOINT CHECK ################

            # Final waypoint gets a slightly tighter stopping distance.
            if (
                self.current_waypoint
                == len(self.route) - 1
            ):

                if distance <= FINAL_TOLERANCE:

                    self.finished = True

                    self.stop_robot()

                    self.get_logger().info(
                        'Final waypoint reached. '
                        f'distance={distance:.3f} m'
                    )

                    return

            else:

                if distance <= WAYPOINT_TOLERANCE:

                    self.get_logger().info(
                        f'Waypoint '
                        f'{self.current_waypoint + 1}/'
                        f'{len(self.route)} reached '
                        f'(distance={distance:.3f} m)'
                    )

                    self.current_waypoint += 1

                    # Reset the smoothed command at a waypoint
                    # so the next segment starts cleanly.
                    self.previous_linear = 0.0
                    self.previous_angular = 0.0

                    return


            ################ TARGET HEADING ################

            target_heading = math.atan2(
                error_y,
                error_x
            )

            heading_error = self.wrap_angle(
                target_heading - yaw
            )


            ################ PATH CONTROL ################

            angular_cmd = (
                KP_HEADING
                * heading_error
            )

            angular_cmd = self.clamp(
                angular_cmd,
                -max_angular_rps,
                max_angular_rps
            )


            # Reduce forward velocity when the robot is
            # facing away from the target.
            heading_factor = math.cos(
                heading_error
            )

            heading_factor = max(
                0.0,
                heading_factor
            )


            if abs(heading_error) > ROTATE_ONLY_ANGLE:

                linear_cmd = 0.0

            else:

                linear_cmd = (
                    0.40
                    * heading_factor
                )


            # Slow down near the waypoint.
            if distance < 0.80:

                distance_factor = self.clamp(
                    distance / 0.80,
                    0.25,
                    1.0
                )

                linear_cmd *= distance_factor


            ################ OBSTACLE CONTROL ################

            (
                obstacle_scale,
                obstacle_turn,
                obstacle_detected
            ) = self.obstacle_avoidance()


            linear_cmd *= obstacle_scale

            angular_cmd += obstacle_turn


            angular_cmd = self.clamp(
                angular_cmd,
                -max_angular_rps,
                max_angular_rps
            )


            ################ CLOSE OBSTACLE CHECK ################

            front, _, _ = (
                self.get_lidar_obstacle_info()
            )

            if front is not None:

                if front <= OBSTACLE_STOP_DISTANCE:

                    # Do not continue driving forward into
                    # a very close obstacle.
                    linear_cmd = 0.0

                    # Continue turning toward free space.
                    if angular_cmd > 0.0:

                        angular_cmd = max(
                            angular_cmd,
                            0.35
                        )

                    elif angular_cmd < 0.0:

                        angular_cmd = min(
                            angular_cmd,
                            -0.35
                        )

                    else:

                        angular_cmd = 0.35


            ################ LOGGING ################

            if (
                self.current_waypoint
                != self.last_logged_waypoint
            ):

                self.last_logged_waypoint = (
                    self.current_waypoint
                )

                self.get_logger().info(
                    f'Target waypoint '
                    f'{self.current_waypoint + 1}/'
                    f'{len(self.route)}'
                )


            if (
                obstacle_detected
                and not self.last_obstacle_state
            ):

                self.get_logger().info(
                    'Obstacle detected; '
                    'avoidance enabled'
                )


            self.last_obstacle_state = (
                obstacle_detected
            )


            ################ PUBLISH ################

            self.publish_command(
                linear_cmd,
                angular_cmd
            )


        except Exception as error:

            # Do not allow an exception to kill the controller
            # while leaving a previous non-zero velocity active.
            self.get_logger().error(
                f'Navigation error: {error}'
            )

            self.stop_robot()

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_nav class to perform its task
    '''

    rclpy.init(args=sys.argv)

    # Kept in the same form as the supplied boilerplate.
    node = rclpy.create_node(
        'ebot_nav_process'
    )

    node.get_logger().info(
        'Node created: eBot navigation process'
    )

    ebot_nav_class = ebot_nav()

    try:

        rclpy.spin(
            ebot_nav_class
        )

    except KeyboardInterrupt:

        pass

    finally:

        ebot_nav_class.stop_robot()

        ebot_nav_class.destroy_node()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
