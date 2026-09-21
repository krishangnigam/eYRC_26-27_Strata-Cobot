#!/usr/bin/env python3

import math
import sys

import rclpy
from rclpy.node import Node

from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32


# ============================================================
# Task 1B constants
# ============================================================

waypoints = [
    (-0.4085, -0.5379, 0.1967),   # 1
    (-0.8000, -0.0005, 0.3967),   # 2
    (-0.7430,  0.5280, 0.1967),   # 3
    (-0.4097,  0.5280, 0.1967),   # 4
    (-0.0763,  0.5280, 0.1967),   # 5
]

servo_ns = '/ur_arm_controller'

twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'

base_frame = 'base_link'

joint_names = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Servo command limits from the supplied Task 1B boilerplate.
cap_linear_mps = 0.15
cap_angular_rps = 0.35

# Additional acceleration limiting.
linear_accel_mps2 = 0.50
angular_accel_rps2 = 1.00

command_timeout_s = 0.15

# Waypoint acceptance tolerances.
position_tolerance_m = 0.008
orientation_tolerance_rad = math.radians(4.0)

# Required time spent at a waypoint.
hold_time_s = 2.0

# Proportional gains.
kp_position = 0.80
kp_orientation = 1.20


# ============================================================
# Quaternion utilities
# ============================================================

def quat_normalize(q):
    x, y, z, w = q

    n = math.sqrt(x * x + y * y + z * z + w * w)

    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)

    return (
        x / n,
        y / n,
        z / n,
        w / n,
    )


def quat_conjugate(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_multiply(q1, q2):

    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quaternion_error(target_q, current_q):
    """
    Return orientation error as axis * angle.

    target_q and current_q are expressed in the same frame.
    """

    target_q = quat_normalize(target_q)
    current_q = quat_normalize(current_q)

    error_q = quat_multiply(
        target_q,
        quat_conjugate(current_q)
    )

    # q and -q represent the same rotation.
    if error_q[3] < 0.0:
        error_q = tuple(-v for v in error_q)

    x, y, z, w = error_q

    w = max(-1.0, min(1.0, w))

    angle = 2.0 * math.acos(w)

    s = math.sqrt(max(0.0, 1.0 - w * w))

    if s < 1e-8 or angle < 1e-8:
        return 0.0, 0.0, 0.0, 0.0

    axis_x = x / s
    axis_y = y / s
    axis_z = z / s

    return (
        axis_x,
        axis_y,
        axis_z,
        angle,
    )


def vector_norm(values):
    return math.sqrt(sum(v * v for v in values))


def cap_vector(values, maximum):

    magnitude = vector_norm(values)

    if magnitude <= maximum or magnitude < 1e-12:
        return values

    scale = maximum / magnitude

    return tuple(v * scale for v in values)


def rate_limit_vector(desired, previous, maximum_delta):

    difference = tuple(
        desired[i] - previous[i]
        for i in range(len(desired))
    )

    difference_norm = vector_norm(difference)

    if difference_norm <= maximum_delta:
        return desired

    limited_difference = tuple(
        previous[i] + difference[i] * maximum_delta / difference_norm
        for i in range(len(desired))
    )

    return limited_difference


# ============================================================
# Node
# ============================================================

class arm_waypoints(Node):

    def __init__(self):

        super().__init__(
            'arm_waypoints_node',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.Parameter.Type.BOOL,
                    True
                )
            ]
        )

        # ----------------------------------------------------
        # Publishers
        # ----------------------------------------------------

        self.twist_pub = self.create_publisher(
            TwistStamped,
            '/delta_twist_cmds',
            10
        )

        self.joint_pub = self.create_publisher(
            JointJog,
            '/delta_joint_cmds',
            10
        )

        # ----------------------------------------------------
        # Subscribers
        # ----------------------------------------------------

        self.tcp_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_raw',
            self.tcpposecb,
            20
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.jointstatecb,
            50
        )

        self.status_sub = self.create_subscription(
            Int32,
            '/arm_status',
            self.armstatuscb,
            10
        )

        # ----------------------------------------------------
        # Controller switching service
        # ----------------------------------------------------

        self.switch_cli = self.create_client(
            SwitchController,
            f'{servo_ns}/switch_controller'
        )

        # ----------------------------------------------------
        # Control timer
        # ----------------------------------------------------

        self.control_period = 0.05

        self.timer = self.create_timer(
            self.control_period,
            self.process_waypoints
        )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.tcp_pose = None
        self.joint_angles = None
        self.arm_status = None

        self.target_orientation = None

        self.current_waypoint = 0

        self.hold_start = None

        self.completed = False

        self.controller_active = False
        self.switch_future = None

        self.last_switch_attempt = 0.0

        self.previous_linear = (0.0, 0.0, 0.0)
        self.previous_angular = (0.0, 0.0, 0.0)

        self.last_control_time = None

        self.last_log_time = 0.0

        self.get_logger().info(
            'Task 1B arm waypoint controller started'
        )

    # ========================================================
    # TCP pose callback
    # ========================================================

    def tcpposecb(self, data):

        self.tcp_pose = data.pose

        # Capture initial orientation once.
        if self.target_orientation is None:

            self.target_orientation = quat_normalize(
                (
                    data.pose.orientation.x,
                    data.pose.orientation.y,
                    data.pose.orientation.z,
                    data.pose.orientation.w,
                )
            )

            self.get_logger().info(
                'Initial TCP orientation captured'
            )

    # ========================================================
    # Joint callback
    # ========================================================

    def jointstatecb(self, data):

        angles = []

        for name in joint_names:

            if name not in data.name:
                self.joint_angles = None
                return

            index = data.name.index(name)

            if index >= len(data.position):
                self.joint_angles = None
                return

            angles.append(
                data.position[index]
            )

        self.joint_angles = angles

    # ========================================================
    # Arm status callback
    # ========================================================

    def armstatuscb(self, data):

        self.arm_status = data.data

    # ========================================================
    # Publish zero velocity
    # ========================================================

    def publish_zero(self):

        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame

        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        self.twist_pub.publish(msg)

        self.previous_linear = (0.0, 0.0, 0.0)
        self.previous_angular = (0.0, 0.0, 0.0)

    # ========================================================
    # Switch to Cartesian twist controller
    # ========================================================

    def try_activate_twist_controller(self):

        if self.controller_active:
            return

        if self.switch_future is not None:

            if not self.switch_future.done():
                return

            try:
                result = self.switch_future.result()

                if result.ok:

                    self.controller_active = True

                    self.get_logger().info(
                        'delta_twist_controller activated'
                    )

                else:

                    self.get_logger().warn(
                        'Controller switch was rejected'
                    )

            except Exception as exc:

                self.get_logger().warn(
                    f'Controller switch failed: {exc}'
                )

            self.switch_future = None
            return

        now = self.get_clock().now().nanoseconds / 1e9

        if now - self.last_switch_attempt < 1.0:
            return

        self.last_switch_attempt = now

        if not self.switch_cli.service_is_ready():

            return

        req = SwitchController.Request()

        req.activate_controllers = [
            twist_controller
        ]

        req.deactivate_controllers = [
            joint_controller
        ]

        req.strictness = SwitchController.Request.STRICT

        self.switch_future = self.switch_cli.call_async(req)

        self.get_logger().info(
            'Requesting delta_twist_controller...'
        )

    # ========================================================
    # Cartesian command
    # ========================================================

    def publish_twist(self, linear, angular):

        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame

        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])

        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])

        self.twist_pub.publish(msg)

    # ========================================================
    # Main controller
    # ========================================================

    def process_waypoints(self):

        # ----------------------------------------------------
        # Basic readiness
        # ----------------------------------------------------

        if self.tcp_pose is None:
            self.publish_zero()
            return

        if self.joint_angles is None:
            self.publish_zero()
            return

        if self.arm_status is None:
            self.publish_zero()
            return

        # Status 0 is the healthy state specified by the
        # supplied boilerplate.
        if self.arm_status != 0:

            self.publish_zero()

            now = self.get_clock().now().nanoseconds / 1e9

            if now - self.last_log_time > 2.0:

                self.get_logger().warn(
                    f'/arm_status = {self.arm_status}; '
                    'holding zero velocity'
                )

                self.last_log_time = now

            return

        # ----------------------------------------------------
        # Make sure Cartesian controller is active
        # ----------------------------------------------------

        if not self.controller_active:

            self.publish_zero()
            self.try_activate_twist_controller()

            return

        # ----------------------------------------------------
        # Finished
        # ----------------------------------------------------

        if self.completed:

            self.publish_zero()
            return

        # ----------------------------------------------------
        # Simulation dt
        # ----------------------------------------------------

        now = self.get_clock().now().nanoseconds / 1e9

        if self.last_control_time is None:

            dt = self.control_period

        else:

            dt = now - self.last_control_time

            if dt <= 0.0:
                dt = self.control_period

            dt = min(dt, 0.2)

        self.last_control_time = now

        # ----------------------------------------------------
        # Current target
        # ----------------------------------------------------

        tx, ty, tz = waypoints[
            self.current_waypoint
        ]

        current_x = self.tcp_pose.position.x
        current_y = self.tcp_pose.position.y
        current_z = self.tcp_pose.position.z

        # ----------------------------------------------------
        # Position error
        # ----------------------------------------------------

        error = (
            tx - current_x,
            ty - current_y,
            tz - current_z,
        )

        distance = vector_norm(error)

        # ----------------------------------------------------
        # Orientation error
        # ----------------------------------------------------

        current_q = quat_normalize(
            (
                self.tcp_pose.orientation.x,
                self.tcp_pose.orientation.y,
                self.tcp_pose.orientation.z,
                self.tcp_pose.orientation.w,
            )
        )

        _, _, _, orientation_angle = quaternion_error(
            self.target_orientation,
            current_q
        )

        axis_x, axis_y, axis_z, angle = quaternion_error(
            self.target_orientation,
            current_q
        )

        orientation_error = (
            axis_x * angle,
            axis_y * angle,
            axis_z * angle,
        )

        # ----------------------------------------------------
        # Check waypoint tolerance
        # ----------------------------------------------------

        inside_position = (
            distance <= position_tolerance_m
        )

        inside_orientation = (
            orientation_angle <= orientation_tolerance_rad
        )

        if inside_position and inside_orientation:

            # Start the hold timer.
            if self.hold_start is None:

                self.hold_start = now

                self.get_logger().info(
                    f'Waypoint {self.current_waypoint + 1} '
                    'reached; starting 2 s hold'
                )

            hold_elapsed = now - self.hold_start

            # Keep a small correcting command while holding.
            position_cmd = cap_vector(
                tuple(
                    kp_position * e
                    for e in error
                ),
                cap_linear_mps
            )

            orientation_cmd = cap_vector(
                tuple(
                    kp_orientation * e
                    for e in orientation_error
                ),
                cap_angular_rps
            )

            if hold_elapsed >= hold_time_s:

                self.previous_linear = (0.0, 0.0, 0.0)
                self.previous_angular = (0.0, 0.0, 0.0)

                self.current_waypoint += 1
                self.hold_start = None

                self.get_logger().info(
                    f'Waypoint {self.current_waypoint} hold complete'
                )

                if self.current_waypoint >= len(waypoints):

                    self.completed = True

                    self.publish_zero()

                    self.get_logger().info(
                        'TASK 1B COMPLETE - all waypoints reached'
                    )

                    return

        else:

            if self.hold_start is not None:

                self.get_logger().info(
                    f'Waypoint {self.current_waypoint + 1} '
                    'left tolerance; restarting hold'
                )

            self.hold_start = None

            # ------------------------------------------------
            # Position proportional control
            # ------------------------------------------------

            position_cmd = tuple(
                kp_position * e
                for e in error
            )

            position_cmd = cap_vector(
                position_cmd,
                cap_linear_mps
            )

            # ------------------------------------------------
            # Orientation proportional control
            # ------------------------------------------------

            orientation_cmd = tuple(
                kp_orientation * e
                for e in orientation_error
            )

            orientation_cmd = cap_vector(
                orientation_cmd,
                cap_angular_rps
            )

        # ----------------------------------------------------
        # Acceleration limiting
        # ----------------------------------------------------

        max_linear_delta = (
            linear_accel_mps2 * dt
        )

        max_angular_delta = (
            angular_accel_rps2 * dt
        )

        linear_cmd = rate_limit_vector(
            position_cmd,
            self.previous_linear,
            max_linear_delta
        )

        angular_cmd = rate_limit_vector(
            orientation_cmd,
            self.previous_angular,
            max_angular_delta
        )

        # Re-cap after rate limiting.
        linear_cmd = cap_vector(
            linear_cmd,
            cap_linear_mps
        )

        angular_cmd = cap_vector(
            angular_cmd,
            cap_angular_rps
        )

        # ----------------------------------------------------
        # Publish EVERY control tick.
        # ----------------------------------------------------

        self.publish_twist(
            linear_cmd,
            angular_cmd
        )

        self.previous_linear = linear_cmd
        self.previous_angular = angular_cmd

        # ----------------------------------------------------
        # Development logging
        # ----------------------------------------------------

        if now - self.last_log_time > 1.0:

            self.get_logger().info(
                f'WP {self.current_waypoint + 1}/{len(waypoints)} | '
                f'distance={distance:.4f} m | '
                f'orientation={math.degrees(orientation_angle):.2f} deg'
            )

            self.last_log_time = now


# ============================================================
# Main
# ============================================================

def main():

    rclpy.init(args=sys.argv)

    node = arm_waypoints()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.publish_zero()
        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
