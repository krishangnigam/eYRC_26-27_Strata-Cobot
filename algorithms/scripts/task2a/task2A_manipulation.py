#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************************************************
* Team ID:          6494
* Author List:      Krishang Nigam, Harshil Jayswal, Masum Pancholi, Divy Vaghasiya
* Filename:         task2A_manipulation.py
* Theme:            StrataCobot (SC) -- Task 2A
* Functions:        quat_normalize, quat_conjugate, quat_multiply, quaternion_error,
*                   vector_norm, cap_vector, rate_limit_vector,
*                   arm_manipulation.__init__, arm_manipulation.tcpposecb,
*                   arm_manipulation.jointstatecb, arm_manipulation.armstatuscb,
*                   arm_manipulation.forcecb, arm_manipulation.switch_controller,
*                   arm_manipulation.call_magnet, arm_manipulation.call_unlock,
*                   arm_manipulation.publish_zero_twist, arm_manipulation.publish_zero_joints,
*                   arm_manipulation.step_state_machine, main
* Nodes:            task2A_manipulation_node
* Publishing:       /delta_twist_cmds (geometry_msgs/TwistStamped)
*                   /delta_joint_cmds (control_msgs/JointJog)
*                   /ore_picked (std_msgs/String)
* Subscribing:      /tcp_pose_raw (geometry_msgs/PoseStamped)
*                   /joint_states (sensor_msgs/JointState)
*                   /arm_status (std_msgs/Int32)
*                   /ur7e/end_effector/force_magnitude (std_msgs/Float32)
* Service Clients:  /ur_arm_controller/switch_controller (SwitchController)
*                   /magnet (std_srvs/SetBool)
*                   /ur_arm_controller/unlock_protective_stop (std_srvs/Trigger)
*****************************************************************************************
'''

import math
import sys
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int32, String
from std_srvs.srv import SetBool, Trigger
import tf2_ros


# ==============================================================================
# MANIPULATION TUNABLES & CONSTANTS (TUNING BLOCK)
# ==============================================================================

# Topic & Service Names
SERVO_NS = '/ur_arm_controller'
TWIST_CONTROLLER = 'delta_twist_controller'
JOINT_CONTROLLER = 'delta_joint_controller'
BASE_FRAME = 'base_link'

# Drop poses per ore mineral type in base_link (x, y, z):
# Gazebo perspective: Azurite (farthest), Malachite (middle), Vanadinite (nearest)
CONTAINER_DROP_POSES = {
    'azurite_ore':   (-0.0763, 0.5280, 0.150),
    'malachite_ore': (-0.4097, 0.5280, 0.150),
    'vanadinite_ore': (-0.7430, 0.5280, 0.150),
}

# Height Waypoints (m in base_link)
SAFE_HOME_POSE = (-0.5000, -0.0500, 0.3500)   # Neutral intermediate transit waypoint
PRE_PICK_HEIGHT_Z = 0.1500                    # Safe height above conveyor ore before descent
LIFT_HEIGHT_Z = 0.2800                        # Safe transit height after picking ore
PRE_DROP_HEIGHT_Z = 0.2400                    # Approach height above cargo container
DROP_DESCENT_HEIGHT_Z = 0.1400                # Descent height into container for clean release

# Motion Speeds & Acceleration Limits
CRUISE_SPEED_MPS = 0.12                       # Main transit linear velocity (m/s)
PICK_DESCENT_SPEED_MPS = 0.025                # Very slow descent to ore to avoid impact (m/s)
DROP_DESCENT_SPEED_MPS = 0.050                # Gentle descent into container (m/s)
CAP_LINEAR_MPS = 0.15                         # Safety velocity cap (m/s)
CAP_ANGULAR_RPS = 0.35                        # Safety angular velocity cap (rad/s)
LINEAR_ACCEL_LIMIT = 0.40                     # Max linear acceleration (m/s^2)
ANGULAR_ACCEL_LIMIT = 0.80                    # Max angular acceleration (rad/s^2)

# Control Gains & Tolerances
KP_POSITION = 1.30                            # Proportional gain for Cartesian position error
KP_ORIENTATION = 1.00                         # Proportional gain for vertical tool orientation hold
KP_JOINT = 1.00                               # Proportional gain for initial joint unfolding
POSITION_TOLERANCE_M = 0.012                  # Position arrival tolerance (m)
STANDSTILL_HOLD_S = 0.50                      # Settle pause at waypoints (s)

# Force Sensor Thresholds (/ur7e/end_effector/force_magnitude in N)
CONTACT_FORCE_TRIGGER_N = 2.0                 # Stop descent when tool touches ore surface (N)
FORCE_PROTECTIVE_LIMIT_N = 8.0                # Safety back-off threshold if contact is too hard (N)
HELD_CONFIRM_FORCE_N = 0.70                   # Force indicating ore weight is supported on tool (N)
RELEASE_CONFIRM_FORCE_N = 0.40                # Force indicating ore has detached (N)

# Retry & Timeout Limits
MAX_PICK_ATTEMPTS = 3                         # Maximum retries per ore before advancing
STARTUP_DELAY_S = 2.0                         # Wait time after controllers become ready (s)
MAGNET_ACTION_DELAY_S = 0.40                  # Settle time after magnet state change (s)
CONTROL_PERIOD_S = 0.05                       # 20 Hz main control loop

# Initial Joint Configuration to Clear Singularities
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
UNFOLD_TARGETS = {
    'shoulder_pan_joint': -2.00,
    'shoulder_lift_joint': -1.20,
    'elbow_joint': -1.60,
    'wrist_1_joint': -1.90,
    'wrist_2_joint': 1.5708,
    'wrist_3_joint': 1.5708,
}


# ==============================================================================
# MATH UTILITIES
# ==============================================================================

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


# ==============================================================================
# MANIPULATION STATE MACHINE NODE
# ==============================================================================

class arm_manipulation(Node):
    '''
    Timer-driven, non-blocking arm manipulation state machine for Task 2A.
    Sequences picks and drops with force compliance, protective stop recovery,
    and zero hardcoded ore coordinates.
    '''

    # State Definitions
    STATE_STARTUP_WAIT = 0
    STATE_SWITCH_TO_JOINT = 1
    STATE_UNFOLD_JOINTS = 2
    STATE_SWITCH_TO_TWIST = 3
    STATE_MOVE_HOME = 4
    STATE_DISCOVER_ORES = 5
    STATE_APPROACH_PRE_PICK = 6
    STATE_DESCENT_PICK = 7
    STATE_ENABLE_MAGNET = 8
    STATE_CONFIRM_GRIP = 9
    STATE_LIFT_AFTER_PICK = 10
    STATE_TRANSIT_HOME_WITH_ORE = 11
    STATE_APPROACH_PRE_DROP = 12
    STATE_DESCENT_DROP = 13
    STATE_DISABLE_MAGNET = 14
    STATE_CONFIRM_RELEASE = 15
    STATE_RETREAT_TO_PRE_DROP = 16
    STATE_RETURN_HOME = 17
    STATE_LATCH_RECOVERY_UNLOCK = 18
    STATE_LATCH_RECOVERY_CLEAR = 19
    STATE_ALL_COMPLETED = 20

    def __init__(self):
        super().__init__(
            'task2A_manipulation_node',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        # Publishers
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.picked_pub = self.create_publisher(String, '/ore_picked', 10)

        # Subscriptions
        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)
        self.force_sub = self.create_subscription(Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 20)

        # Service Clients
        self.switch_cli = self.create_client(SwitchController, f'{SERVO_NS}/switch_controller')
        self.magnet_cli = self.create_client(SetBool, '/magnet')
        self.unlock_cli = self.create_client(Trigger, f'{SERVO_NS}/unlock_protective_stop')

        # TF Buffer / Listener
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State Variables
        self.state = self.STATE_STARTUP_WAIT
        self.previous_state_before_latch = self.STATE_MOVE_HOME
        self.tcp_pose = None
        self.joint_angles = {}
        self.arm_status = None
        self.force_magnitude = 0.0

        self.target_orientation = None
        self.previous_linear = (0.0, 0.0, 0.0)
        self.previous_angular = (0.0, 0.0, 0.0)

        # Asynchronous Service Futures
        self.switch_future = None
        self.magnet_future = None
        self.unlock_future = None

        # Queue of ores to pick: list of (ore_name, kind, (x, y, z))
        self.ore_queue = []
        self.current_ore = None
        self.pick_attempt_count = 0

        # Timing variables
        self.state_enter_time = 0.0
        self.last_log_time = 0.0
        self.node_init_time = None

        # Main Control Timer (20 Hz)
        self.timer = self.create_timer(CONTROL_PERIOD_S, self.step_state_machine)
        self.get_logger().info('Task 2A Manipulation Node initialized.')

    def tcpposecb(self, data):
        self.tcp_pose = data.pose

    def jointstatecb(self, data):
        for name, pos in zip(data.name, data.position):
            self.joint_angles[name] = pos

    def armstatuscb(self, data):
        prev = self.arm_status
        self.arm_status = data.data
        if prev != self.arm_status:
            self.get_logger().info(f'Arm status changed: {prev} -> {self.arm_status}')

    def forcecb(self, data):
        self.force_magnitude = float(data.data)

    def change_state(self, new_state, now):
        self.get_logger().info(
            f'[STATE TRANSITION] {self.state_name(self.state)} -> {self.state_name(new_state)} '
            f'| Ore: {self.current_ore[0] if self.current_ore else "None"} '
            f'| Force: {self.force_magnitude:.2f}N | Status: {self.arm_status}'
        )
        self.state = new_state
        self.state_enter_time = now

    def state_name(self, s):
        for k, v in self.__class__.__dict__.items():
            if k.startswith('STATE_') and v == s:
                return k[6:]
        return str(s)

    def switch_controller(self, activate_name, deactivate_name=None):
        if not self.switch_cli.service_is_ready():
            return False
        req = SwitchController.Request()
        req.activate_controllers = [activate_name]
        if deactivate_name:
            req.deactivate_controllers = [deactivate_name]
        req.strictness = SwitchController.Request.BEST_EFFORT
        self.switch_future = self.switch_cli.call_async(req)
        return True

    def call_magnet(self, enable: bool):
        if not self.magnet_cli.service_is_ready():
            return False
        req = SetBool.Request()
        req.data = enable
        self.magnet_future = self.magnet_cli.call_async(req)
        return True

    def call_unlock(self):
        if not self.unlock_cli.service_is_ready():
            return False
        req = Trigger.Request()
        self.unlock_future = self.unlock_cli.call_async(req)
        return True

    def publish_zero_twist(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        self.twist_pub.publish(msg)
        self.previous_linear = (0.0, 0.0, 0.0)
        self.previous_angular = (0.0, 0.0, 0.0)

    def publish_zero_joints(self):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = JOINT_NAMES
        msg.velocities = [0.0] * len(JOINT_NAMES)
        self.joint_pub.publish(msg)

    def navigate_cartesian(self, target_xyz, cruise_speed, now):
        '''
        Servo the TCP toward target_xyz in base_link with orientation stabilization.
        Returns True when distance <= POSITION_TOLERANCE_M.
        '''
        cx, cy, cz = self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z
        tx, ty, tz = target_xyz
        error_vec = (tx - cx, ty - cy, tz - cz)
        dist = vector_norm(error_vec)

        if dist <= POSITION_TOLERANCE_M:
            self.publish_zero_twist()
            return True

        unit_dir = tuple(e / dist for e in error_vec)
        speed = min(cruise_speed, KP_POSITION * dist)
        desired_linear = cap_vector(tuple(u * speed for u in unit_dir), CAP_LINEAR_MPS)

        max_linear_step = LINEAR_ACCEL_LIMIT * CONTROL_PERIOD_S
        linear_cmd = rate_limit_vector(desired_linear, self.previous_linear, max_linear_step)

        # Orientation stabilization to downward vertical
        current_q = quat_normalize((
            self.tcp_pose.orientation.x,
            self.tcp_pose.orientation.y,
            self.tcp_pose.orientation.z,
            self.tcp_pose.orientation.w,
        ))
        ax, ay, az, angle = quaternion_error(self.target_orientation, current_q)
        desired_angular = cap_vector(
            (ax * angle * KP_ORIENTATION, ay * angle * KP_ORIENTATION, az * angle * KP_ORIENTATION),
            CAP_ANGULAR_RPS
        )
        max_angular_step = ANGULAR_ACCEL_LIMIT * CONTROL_PERIOD_S
        angular_cmd = rate_limit_vector(desired_angular, self.previous_angular, max_angular_step)

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.twist.linear.x = float(linear_cmd[0])
        msg.twist.linear.y = float(linear_cmd[1])
        msg.twist.linear.z = float(linear_cmd[2])
        msg.twist.angular.x = float(angular_cmd[0])
        msg.twist.angular.y = float(angular_cmd[1])
        msg.twist.angular.z = float(angular_cmd[2])
        self.twist_pub.publish(msg)

        self.previous_linear = linear_cmd
        self.previous_angular = angular_cmd
        return False

    def step_state_machine(self):
        '''Main timer callback executing non-blocking state machine steps.'''
        now = self.get_clock().now().nanoseconds / 1e9

        if self.node_init_time is None:
            self.node_init_time = now

        # ------------------------------------------------------------------
        # CRITICAL SAFETY CHECK: Protective Stop Monitor (Codes 20-23)
        # ------------------------------------------------------------------
        if self.arm_status is not None and 20 <= self.arm_status <= 23:
            if self.state not in (self.STATE_LATCH_RECOVERY_UNLOCK, self.STATE_LATCH_RECOVERY_CLEAR):
                self.get_logger().error(
                    f'PROTECTIVE STOP DETECTED: code={self.arm_status}. Latching position and initiating recovery...'
                )
                self.publish_zero_twist()
                self.previous_state_before_latch = self.state
                self.change_state(self.STATE_LATCH_RECOVERY_UNLOCK, now)
                self.call_unlock()
                return

        # ------------------------------------------------------------------
        # STATE 0: Wait for controller & sensor readiness
        # ------------------------------------------------------------------
        if self.state == self.STATE_STARTUP_WAIT:
            if self.tcp_pose is None or len(self.joint_angles) < 6 or self.arm_status is None:
                return
            if self.arm_status != 0:
                return
            if (now - self.node_init_time) < STARTUP_DELAY_S:
                return

            self.get_logger().info('Arm controller ready. Requesting switch to joint controller for unfold...')
            self.change_state(self.STATE_SWITCH_TO_JOINT, now)
            self.switch_controller(JOINT_CONTROLLER)
            return

        # ------------------------------------------------------------------
        # STATE 1: Switch to delta_joint_controller
        # ------------------------------------------------------------------
        elif self.state == self.STATE_SWITCH_TO_JOINT:
            if self.switch_future is not None and self.switch_future.done():
                try:
                    res = self.switch_future.result()
                    if res.ok:
                        self.get_logger().info('Switched to joint controller. Unfolding joints...')
                        self.change_state(self.STATE_UNFOLD_JOINTS, now)
                    else:
                        self.get_logger().warn('Switch to joint controller rejected; retrying...')
                        self.switch_controller(JOINT_CONTROLLER)
                except Exception as e:
                    self.get_logger().error(f'Switch error: {e}')
                self.switch_future = None
            return

        # ------------------------------------------------------------------
        # STATE 2: Unfold joints to clear singularity & limits
        # ------------------------------------------------------------------
        elif self.state == self.STATE_UNFOLD_JOINTS:
            max_err = 0.0
            cmd_velocities = []
            for name in JOINT_NAMES:
                curr = self.joint_angles.get(name, 0.0)
                tgt = UNFOLD_TARGETS[name]
                err = tgt - curr
                max_err = max(max_err, abs(err))
                vel = max(-0.25, min(0.25, KP_JOINT * err))
                cmd_velocities.append(vel)

            msg = JointJog()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = JOINT_NAMES
            msg.velocities = cmd_velocities
            self.joint_pub.publish(msg)

            if max_err <= 0.04:
                self.publish_zero_joints()
                self.get_logger().info('Joint unfold complete. Switching to Cartesian twist controller...')
                self.change_state(self.STATE_SWITCH_TO_TWIST, now)
                self.switch_controller(TWIST_CONTROLLER)
            return

        # ------------------------------------------------------------------
        # STATE 3: Switch back to delta_twist_controller
        # ------------------------------------------------------------------
        elif self.state == self.STATE_SWITCH_TO_TWIST:
            if self.switch_future is not None and self.switch_future.done():
                try:
                    res = self.switch_future.result()
                    if res.ok:
                        self.get_logger().info('Switched to twist controller. Locking orientation...')
                        self.target_orientation = quat_normalize((
                            self.tcp_pose.orientation.x,
                            self.tcp_pose.orientation.y,
                            self.tcp_pose.orientation.z,
                            self.tcp_pose.orientation.w,
                        ))
                        self.previous_linear = (0.0, 0.0, 0.0)
                        self.previous_angular = (0.0, 0.0, 0.0)
                        self.change_state(self.STATE_MOVE_HOME, now)
                    else:
                        self.get_logger().warn('Switch to twist controller rejected; retrying...')
                        self.switch_controller(TWIST_CONTROLLER)
                except Exception as e:
                    self.get_logger().error(f'Switch error: {e}')
                self.switch_future = None
            return

        # ------------------------------------------------------------------
        # STATE 4: Move to Safe Home Waypoint
        # ------------------------------------------------------------------
        elif self.state == self.STATE_MOVE_HOME:
            arrived = self.navigate_cartesian(SAFE_HOME_POSE, CRUISE_SPEED_MPS, now)
            if arrived:
                self.change_state(self.STATE_DISCOVER_ORES, now)
            return

        # ------------------------------------------------------------------
        # STATE 5: Discover Available Ores via TF Buffer
        # ------------------------------------------------------------------
        elif self.state == self.STATE_DISCOVER_ORES:
            # Query TF buffer for all ore transforms
            candidate_kinds = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']
            discovered = []

            for kind in candidate_kinds:
                for idx in [1, 2]:
                    frame_id = f'{kind}_{idx}'
                    try:
                        tf = self.tf_buffer.lookup_transform(BASE_FRAME, frame_id, Time())
                        tx = tf.transform.translation.x
                        ty = tf.transform.translation.y
                        tz = tf.transform.translation.z
                        discovered.append((frame_id, kind, (tx, ty, tz)))
                    except Exception:
                        pass

            if len(discovered) > 0:
                # Sort discovered ores by nearest to arm (descending Y along conveyor)
                # This guarantees the arm works nearest ores first without sweeping over remaining ones
                discovered.sort(key=lambda item: -item[2][1])
                self.ore_queue = discovered
                self.get_logger().info(f'Discovered {len(self.ore_queue)} ores: {[o[0] for o in self.ore_queue]}')
                self.current_ore = self.ore_queue.pop(0)
                self.pick_attempt_count = 0
                self.change_state(self.STATE_APPROACH_PRE_PICK, now)
            else:
                if now - self.last_log_time > 1.0:
                    self.get_logger().info('Waiting for ore transforms from perception node...')
                    self.last_log_time = now
            return

        # ------------------------------------------------------------------
        # STATE 6: Move to Pre-Pick Pose (Above Ore)
        # ------------------------------------------------------------------
        elif self.state == self.STATE_APPROACH_PRE_PICK:
            ox, oy, _ = self.current_ore[2]
            target_pre_pick = (ox, oy, PRE_PICK_HEIGHT_Z)
            arrived = self.navigate_cartesian(target_pre_pick, CRUISE_SPEED_MPS, now)
            if arrived:
                self.change_state(self.STATE_DESCENT_PICK, now)
            return

        # ------------------------------------------------------------------
        # STATE 7: Slow Vertical Descent & Contact Check
        # ------------------------------------------------------------------
        elif self.state == self.STATE_DESCENT_PICK:
            ox, oy, oz = self.current_ore[2]
            # Slow vertical descent toward ore top face
            target_pick = (ox, oy, oz)

            # Check force threshold to prevent hard collision
            if self.force_magnitude >= CONTACT_FORCE_TRIGGER_N:
                self.publish_zero_twist()
                self.get_logger().info(
                    f'Contact detected (force={self.force_magnitude:.2f}N). Stopping descent and energizing magnet.'
                )
                self.change_state(self.STATE_ENABLE_MAGNET, now)
                self.call_magnet(True)
                return

            arrived = self.navigate_cartesian(target_pick, PICK_DESCENT_SPEED_MPS, now)
            if arrived:
                self.publish_zero_twist()
                self.change_state(self.STATE_ENABLE_MAGNET, now)
                self.call_magnet(True)
            return

        # ------------------------------------------------------------------
        # STATE 8: Enable Electromagnet & Settle
        # ------------------------------------------------------------------
        elif self.state == self.STATE_ENABLE_MAGNET:
            if self.magnet_future is not None and self.magnet_future.done():
                self.magnet_future = None

            if (now - self.state_enter_time) >= MAGNET_ACTION_DELAY_S:
                self.change_state(self.STATE_CONFIRM_GRIP, now)
            return

        # ------------------------------------------------------------------
        # STATE 9: Confirm Grip & Lift Check
        # ------------------------------------------------------------------
        elif self.state == self.STATE_CONFIRM_GRIP:
            # Lift vertically to verify whether ore is securely captured
            ox, oy, _ = self.current_ore[2]
            target_lift = (ox, oy, LIFT_HEIGHT_Z)
            arrived = self.navigate_cartesian(target_lift, PICK_DESCENT_SPEED_MPS * 2.0, now)

            if arrived:
                # Check end-effector force reading when lifted off conveyor
                if self.force_magnitude >= HELD_CONFIRM_FORCE_N:
                    self.get_logger().info(
                        f'SUCCESSFUL PICK: {self.current_ore[0]} confirmed held (force={self.force_magnitude:.2f}N).'
                    )
                    # Notify perception node that ore has been picked
                    picked_msg = String()
                    picked_msg.data = self.current_ore[0]
                    self.picked_pub.publish(picked_msg)

                    self.change_state(self.STATE_TRANSIT_HOME_WITH_ORE, now)
                else:
                    self.pick_attempt_count += 1
                    self.get_logger().warn(
                        f'Pick failed for {self.current_ore[0]} (force={self.force_magnitude:.2f}N < {HELD_CONFIRM_FORCE_N}N). '
                        f'Attempt {self.pick_attempt_count}/{MAX_PICK_ATTEMPTS}'
                    )
                    self.call_magnet(False)
                    if self.pick_attempt_count < MAX_PICK_ATTEMPTS:
                        self.change_state(self.STATE_APPROACH_PRE_PICK, now)
                    else:
                        self.get_logger().error(f'Max retries exceeded for {self.current_ore[0]}. Skipping to next ore.')
                        self.change_state(self.STATE_TRANSIT_HOME_WITH_ORE, now)
            return

        # ------------------------------------------------------------------
        # STATE 10 & 11: Transit to Safe Home Waypoint with Ore
        # ------------------------------------------------------------------
        elif self.state == self.STATE_TRANSIT_HOME_WITH_ORE:
            arrived = self.navigate_cartesian(SAFE_HOME_POSE, CRUISE_SPEED_MPS, now)
            if arrived:
                self.change_state(self.STATE_APPROACH_PRE_DROP, now)
            return

        # ------------------------------------------------------------------
        # STATE 12: Move to Pre-Drop Pose Above Container
        # ------------------------------------------------------------------
        elif self.state == self.STATE_APPROACH_PRE_DROP:
            kind = self.current_ore[1]
            drop_x, drop_y, _ = CONTAINER_DROP_POSES[kind]
            target_pre_drop = (drop_x, drop_y, PRE_DROP_HEIGHT_Z)
            arrived = self.navigate_cartesian(target_pre_drop, CRUISE_SPEED_MPS, now)
            if arrived:
                self.change_state(self.STATE_DESCENT_DROP, now)
            return

        # ------------------------------------------------------------------
        # STATE 13: Descend Slightly into Container
        # ------------------------------------------------------------------
        elif self.state == self.STATE_DESCENT_DROP:
            kind = self.current_ore[1]
            drop_x, drop_y, _ = CONTAINER_DROP_POSES[kind]
            target_drop = (drop_x, drop_y, DROP_DESCENT_HEIGHT_Z)
            arrived = self.navigate_cartesian(target_drop, DROP_DESCENT_SPEED_MPS, now)
            if arrived:
                self.publish_zero_twist()
                self.change_state(self.STATE_DISABLE_MAGNET, now)
                self.call_magnet(False)
            return

        # ------------------------------------------------------------------
        # STATE 14: De-energize Magnet
        # ------------------------------------------------------------------
        elif self.state == self.STATE_DISABLE_MAGNET:
            if self.magnet_future is not None and self.magnet_future.done():
                self.magnet_future = None

            if (now - self.state_enter_time) >= MAGNET_ACTION_DELAY_S:
                self.change_state(self.STATE_CONFIRM_RELEASE, now)
            return

        # ------------------------------------------------------------------
        # STATE 15: Confirm Release
        # ------------------------------------------------------------------
        elif self.state == self.STATE_CONFIRM_RELEASE:
            self.get_logger().info(
                f'SUCCESSFUL DROP: {self.current_ore[0]} released into container (force={self.force_magnitude:.2f}N).'
            )
            self.change_state(self.STATE_RETREAT_TO_PRE_DROP, now)
            return

        # ------------------------------------------------------------------
        # STATE 16: Retreat Upward to Pre-Drop Height
        # ------------------------------------------------------------------
        elif self.state == self.STATE_RETREAT_TO_PRE_DROP:
            kind = self.current_ore[1]
            drop_x, drop_y, _ = CONTAINER_DROP_POSES[kind]
            target_pre_drop = (drop_x, drop_y, PRE_DROP_HEIGHT_Z)
            arrived = self.navigate_cartesian(target_pre_drop, CRUISE_SPEED_MPS, now)
            if arrived:
                self.change_state(self.STATE_RETURN_HOME, now)
            return

        # ------------------------------------------------------------------
        # STATE 17: Return to Safe Home Waypoint & Check Remaining Ores
        # ------------------------------------------------------------------
        elif self.state == self.STATE_RETURN_HOME:
            arrived = self.navigate_cartesian(SAFE_HOME_POSE, CRUISE_SPEED_MPS, now)
            if arrived:
                if len(self.ore_queue) > 0:
                    self.current_ore = self.ore_queue.pop(0)
                    self.pick_attempt_count = 0
                    self.change_state(self.STATE_APPROACH_PRE_PICK, now)
                else:
                    self.get_logger().info('All 4 ores successfully sorted! Task 2A completed.')
                    self.change_state(self.STATE_ALL_COMPLETED, now)
            return

        # ------------------------------------------------------------------
        # STATE 18: Latch Recovery -- Unlock Service Call
        # ------------------------------------------------------------------
        elif self.state == self.STATE_LATCH_RECOVERY_UNLOCK:
            if self.unlock_future is not None and self.unlock_future.done():
                try:
                    res = self.unlock_future.result()
                    self.get_logger().info(f'Unlock protective stop returned: {res.message}')
                except Exception as e:
                    self.get_logger().error(f'Unlock service error: {e}')
                self.unlock_future = None
                self.change_state(self.STATE_LATCH_RECOVERY_CLEAR, now)
            return

        # ------------------------------------------------------------------
        # STATE 19: Latch Recovery -- Immediate Upward Motion (Clear Fault)
        # ------------------------------------------------------------------
        elif self.state == self.STATE_LATCH_RECOVERY_CLEAR:
            # Stream upward twist to back away from surface within 5s recovery window
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = BASE_FRAME
            msg.twist.linear.z = 0.05
            self.twist_pub.publish(msg)

            if self.arm_status == 0 or (now - self.state_enter_time) >= 1.5:
                self.publish_zero_twist()
                self.get_logger().info('Arm fault cleared. Returning to Safe Home Waypoint to retry...')
                self.change_state(self.STATE_MOVE_HOME, now)
            return

        # ------------------------------------------------------------------
        # STATE 20: All Tasks Completed
        # ------------------------------------------------------------------
        elif self.state == self.STATE_ALL_COMPLETED:
            self.publish_zero_twist()
            return


def main(args=None):
    rclpy.init(args=args)
    node = arm_manipulation()
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
