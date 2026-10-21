#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1B of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task1b.py
# Functions:        quat_normalize, quat_conjugate, quat_multiply, quaternion_error,
#                   vector_norm, cap_vector, rate_limit_vector, arm_waypoints.__init__,
#                   arm_waypoints.tcpposecb, arm_waypoints.jointstatecb,
#                   arm_waypoints.armstatuscb, arm_waypoints.switch_controller,
#                   arm_waypoints.publish_zero_twist, arm_waypoints.publish_zero_joints,
#                   arm_waypoints.process_waypoints, main
# Nodes:            arm_waypoints_node
#
# Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds ]
# Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status ]


################### IMPORT MODULES #######################

import math
import sys

import rclpy
from rclpy.node import Node

from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32


##################### TASK CONSTANTS #######################

# Tool positions in base_link, in metres, in the order they must be reached.
waypoints = [
    (-0.4085, -0.5379, 0.1967),   # Waypoint 1
    (-0.8000, -0.0005, 0.3967),   # Waypoint 2
    (-0.7430,  0.5280, 0.1967),   # Waypoint 3
    (-0.4097,  0.5280, 0.1967),   # Waypoint 4
    (-0.0763,  0.5280, 0.1967),   # Waypoint 5
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

# Physical and servo controller limits
cap_linear_mps = 0.15      # Max linear velocity magnitude (m/s)
cap_angular_rps = 0.35     # Max angular velocity magnitude (rad/s)
cap_joint_rps = 0.35       # Max joint velocity magnitude (rad/s)

# Smooth acceleration limiting to eliminate sudden step-changes
linear_accel_limit = 0.40  # m/s^2
angular_accel_limit = 0.80 # rad/s^2

# Navigation tuning
cruise_speed = 0.10        # Operating cruise speed (m/s)
kp_position = 1.30         # Proportional gain for position error
kp_orientation = 1.00      # Proportional gain for orientation stabilization
kp_joint = 1.00            # Proportional gain for joint unfolding

position_tolerance_m = 0.010   # 10 mm arrival tolerance
hold_time_s = 2.25             # 2.25 s hold to strictly exceed 2.0 s requirement

# Open working configuration for joint-space unfolding
unfold_targets = {
    'shoulder_pan_joint': -2.00,
    'shoulder_lift_joint': -1.20,
    'elbow_joint': -1.60,
    'wrist_1_joint': -1.90,
    'wrist_2_joint': 1.5708,
    'wrist_3_joint': 1.5708,
}


##################### MATH UTILITIES #######################

def quat_normalize(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


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
    target_q = quat_normalize(target_q)
    current_q = quat_normalize(current_q)

    error_q = quat_multiply(target_q, quat_conjugate(current_q))

    # q and -q represent identical rotations
    if error_q[3] < 0.0:
        error_q = tuple(-v for v in error_q)

    x, y, z, w = error_q
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))

    if s < 1e-8 or angle < 1e-8:
        return 0.0, 0.0, 0.0, 0.0

    return (x / s, y / s, z / s, angle)


def vector_norm(values):
    return math.sqrt(sum(v * v for v in values))


def cap_vector(values, maximum):
    magnitude = vector_norm(values)
    if magnitude <= maximum or magnitude < 1e-12:
        return values
    scale = maximum / magnitude
    return tuple(v * scale for v in values)


def rate_limit_vector(desired, previous, max_delta):
    diff = tuple(d - p for d, p in zip(desired, previous))
    norm = vector_norm(diff)
    if norm <= max_delta:
        return desired
    scale = max_delta / norm
    return tuple(p + d_i * scale for p, d_i in zip(previous, diff))


##################### CLASS DEFINITION #######################

class arm_waypoints(Node):
    '''
    ___CLASS___

    Description:    Drives the UR7e tool center point sequentially through 5 waypoints
                    using joint-space unfolding, Cartesian twist servoing, and exact
                    standstill holds.
    '''

    # State machine definitions
    STATE_WAIT_FOR_READY = 0
    STATE_SWITCH_TO_JOINT = 1
    STATE_UNFOLD_JOINTS = 2
    STATE_SWITCH_TO_TWIST = 3
    STATE_NAVIGATE = 4
    STATE_HOLD = 5
    STATE_COMPLETED = 6

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

        ############ Topic PUBLISHERS ############
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        ############ Topic SUBSCRIPTIONS ############
        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)

        ############ Service CLIENTS ############
        self.switch_cli = self.create_client(SwitchController, f'{servo_ns}/switch_controller')

        ############ State Variables ############
        self.tcp_pose = None
        self.joint_angles = {}
        self.arm_status = None

        self.state = self.STATE_WAIT_FOR_READY
        self.switch_future = None
        self.last_switch_call = 0.0

        self.current_waypoint = 0
        self.target_orientation = None
        self.hold_start_time = None

        self.previous_linear = (0.0, 0.0, 0.0)
        self.previous_angular = (0.0, 0.0, 0.0)

        self.last_log_time = 0.0

        ############ Control Timer ############
        self.control_period = 0.05   # 20 Hz
        self.timer = self.create_timer(self.control_period, self.process_waypoints)

        self.get_logger().info('Arm waypoints node initialized and running on simulation clock.')

    def tcpposecb(self, data):
        self.tcp_pose = data.pose

    def jointstatecb(self, data):
        for name, pos in zip(data.name, data.position):
            self.joint_angles[name] = pos

    def armstatuscb(self, data):
        self.arm_status = data.data

    def switch_controller(self, activate_name, deactivate_name=None):
        '''
        Asynchronously invoke /ur_arm_controller/switch_controller without blocking the spinner.
        '''
        if not self.switch_cli.service_is_ready():
            return False

        req = SwitchController.Request()
        req.activate_controllers = [activate_name]
        if deactivate_name:
            req.deactivate_controllers = [deactivate_name]
        req.strictness = SwitchController.Request.BEST_EFFORT

        self.switch_future = self.switch_cli.call_async(req)
        return True

    def publish_zero_twist(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        self.twist_pub.publish(msg)
        self.previous_linear = (0.0, 0.0, 0.0)
        self.previous_angular = (0.0, 0.0, 0.0)

    def publish_zero_joints(self):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = [0.0] * len(joint_names)
        self.joint_pub.publish(msg)

    def process_waypoints(self):
        now = self.get_clock().now().nanoseconds / 1e9

        # Ensure sensor data and controller status are ready
        if self.tcp_pose is None or len(self.joint_angles) < 6 or self.arm_status is None:
            return

        # If arm is latched or warning active, hold zero velocity
        if self.arm_status != 0:
            if self.state in (self.STATE_NAVIGATE, self.STATE_HOLD, self.STATE_COMPLETED):
                self.publish_zero_twist()
            elif self.state == self.STATE_UNFOLD_JOINTS:
                self.publish_zero_joints()
            if now - self.last_log_time > 2.0:
                self.get_logger().warn(f'Safety alert: /arm_status = {self.arm_status}; holding position.')
                self.last_log_time = now
            return

        # ------------------------------------------------------------------
        # STATE 0: Wait for ready & trigger switch to joint controller
        # ------------------------------------------------------------------
        if self.state == self.STATE_WAIT_FOR_READY:
            self.state = self.STATE_SWITCH_TO_JOINT
            self.get_logger().info('Arm ready. Requesting switch to delta_joint_controller for unfolding...')
            return

        # ------------------------------------------------------------------
        # STATE 1: Switch to delta_joint_controller
        # ------------------------------------------------------------------
        elif self.state == self.STATE_SWITCH_TO_JOINT:
            if self.switch_future is None:
                if now - self.last_switch_call >= 1.0:
                    self.last_switch_call = now
                    self.switch_controller(joint_controller)
            else:
                if self.switch_future.done():
                    try:
                        res = self.switch_future.result()
                        if res.ok:
                            self.get_logger().info('Switched to delta_joint_controller. Beginning joint unfold.')
                            self.state = self.STATE_UNFOLD_JOINTS
                        else:
                            self.get_logger().warn('Switch to delta_joint_controller was not accepted; retrying...')
                    except Exception as exc:
                        self.get_logger().error(f'Switch controller error: {exc}')
                    self.switch_future = None
            return

        # ------------------------------------------------------------------
        # STATE 2: Unfold joints to clear elbow limit & wrist singularity
        # ------------------------------------------------------------------
        elif self.state == self.STATE_UNFOLD_JOINTS:
            max_joint_err = 0.0
            cmd_velocities = []

            for name in joint_names:
                curr_angle = self.joint_angles.get(name, 0.0)
                target_angle = unfold_targets[name]
                err = target_angle - curr_angle
                max_joint_err = max(max_joint_err, abs(err))

                vel = max(-0.25, min(0.25, kp_joint * err))
                cmd_velocities.append(vel)

            msg = JointJog()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = joint_names
            msg.velocities = cmd_velocities
            self.joint_pub.publish(msg)

            if now - self.last_log_time > 1.5:
                self.get_logger().info(f'Unfolding arm: max joint error = {max_joint_err:.3f} rad')
                self.last_log_time = now

            if max_joint_err <= 0.04:
                self.publish_zero_joints()
                self.get_logger().info('Unfold complete! Requesting switch to delta_twist_controller...')
                self.state = self.STATE_SWITCH_TO_TWIST
            return

        # ------------------------------------------------------------------
        # STATE 3: Switch back to delta_twist_controller
        # ------------------------------------------------------------------
        elif self.state == self.STATE_SWITCH_TO_TWIST:
            if self.switch_future is None:
                if now - self.last_switch_call >= 1.0:
                    self.last_switch_call = now
                    self.switch_controller(twist_controller)
            else:
                if self.switch_future.done():
                    try:
                        res = self.switch_future.result()
                        if res.ok:
                            self.get_logger().info('Switched to delta_twist_controller. Ready for waypoint navigation.')
                            # Lock target orientation from current stable unfolded pose
                            self.target_orientation = quat_normalize((
                                self.tcp_pose.orientation.x,
                                self.tcp_pose.orientation.y,
                                self.tcp_pose.orientation.z,
                                self.tcp_pose.orientation.w,
                            ))
                            self.previous_linear = (0.0, 0.0, 0.0)
                            self.previous_angular = (0.0, 0.0, 0.0)
                            self.state = self.STATE_NAVIGATE
                        else:
                            self.get_logger().warn('Switch to delta_twist_controller was not accepted; retrying...')
                    except Exception as exc:
                        self.get_logger().error(f'Switch controller error: {exc}')
                    self.switch_future = None
            return

        # ------------------------------------------------------------------
        # STATE 4: Navigate to current waypoint
        # ------------------------------------------------------------------
        elif self.state == self.STATE_NAVIGATE:
            tx, ty, tz = waypoints[self.current_waypoint]
            cx, cy, cz = self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z

            error_vec = (tx - cx, ty - cy, tz - cz)
            dist = vector_norm(error_vec)

            # Check arrival at waypoint
            if dist <= position_tolerance_m:
                self.publish_zero_twist()
                self.hold_start_time = now
                self.state = self.STATE_HOLD
                self.get_logger().info(
                    f'Arrived at Waypoint {self.current_waypoint + 1}/{len(waypoints)} '
                    f'(distance = {dist * 1000.0:.1f} mm). Beginning {hold_time_s:.2f} s standstill hold...'
                )
                return

            # Compute proportional velocity with smooth tapering
            unit_dir = tuple(e / dist for e in error_vec)
            speed = min(cruise_speed, kp_position * dist)
            desired_linear = cap_vector(tuple(u * speed for u in unit_dir), cap_linear_mps)

            # Acceleration limiting to avoid abrupt step changes
            max_linear_step = linear_accel_limit * self.control_period
            linear_cmd = rate_limit_vector(desired_linear, self.previous_linear, max_linear_step)

            # Orientation stabilization
            current_q = quat_normalize((
                self.tcp_pose.orientation.x,
                self.tcp_pose.orientation.y,
                self.tcp_pose.orientation.z,
                self.tcp_pose.orientation.w,
            ))
            ax, ay, az, angle = quaternion_error(self.target_orientation, current_q)
            desired_angular = cap_vector(
                (ax * angle * kp_orientation, ay * angle * kp_orientation, az * angle * kp_orientation),
                cap_angular_rps
            )
            max_angular_step = angular_accel_limit * self.control_period
            angular_cmd = rate_limit_vector(desired_angular, self.previous_angular, max_angular_step)

            # Publish twist command
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = base_frame
            msg.twist.linear.x = float(linear_cmd[0])
            msg.twist.linear.y = float(linear_cmd[1])
            msg.twist.linear.z = float(linear_cmd[2])
            msg.twist.angular.x = float(angular_cmd[0])
            msg.twist.angular.y = float(angular_cmd[1])
            msg.twist.angular.z = float(angular_cmd[2])
            self.twist_pub.publish(msg)

            self.previous_linear = linear_cmd
            self.previous_angular = angular_cmd

            if now - self.last_log_time > 1.0:
                self.get_logger().info(
                    f'Navigating WP {self.current_waypoint + 1}/{len(waypoints)} | '
                    f'dist = {dist:.4f} m | speed = {speed:.3f} m/s'
                )
                self.last_log_time = now
            return

        # ------------------------------------------------------------------
        # STATE 5: Exact standstill hold for >= 2.0 seconds
        # ------------------------------------------------------------------
        elif self.state == self.STATE_HOLD:
            # Continuous zero velocity to guarantee zero movement and zero drift
            self.publish_zero_twist()

            # Ensure tool remains within bounds
            tx, ty, tz = waypoints[self.current_waypoint]
            cx, cy, cz = self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z
            dist = vector_norm((tx - cx, ty - cy, tz - cz))

            if dist > 0.035:
                self.get_logger().warn(f'Tool drifted outside hold bounds ({dist:.3f} m); resuming navigation.')
                self.state = self.STATE_NAVIGATE
                self.hold_start_time = None
                return

            elapsed = now - self.hold_start_time
            if elapsed >= hold_time_s:
                self.get_logger().info(
                    f'Waypoint {self.current_waypoint + 1} held successfully for {elapsed:.2f} s.'
                )
                self.current_waypoint += 1
                self.hold_start_time = None

                if self.current_waypoint >= len(waypoints):
                    self.state = self.STATE_COMPLETED
                    self.get_logger().info('*** ALL 5 WAYPOINTS REACHED AND HELD! TASK 1B COMPLETE ***')
                else:
                    self.state = self.STATE_NAVIGATE
            return

        # ------------------------------------------------------------------
        # STATE 6: Completed
        # ------------------------------------------------------------------
        elif self.state == self.STATE_COMPLETED:
            self.publish_zero_twist()
            return


##################### FUNCTION DEFINITION #######################

def main():
    rclpy.init(args=sys.argv)
    node = arm_waypoints()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero_twist()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
