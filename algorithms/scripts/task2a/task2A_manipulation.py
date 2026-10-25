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
import statistics
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
# Computed container centres from Kepler Colony rover collision geometry
CONTAINER_DROP_POSES = {
    'azurite_ore':   (-0.0281, 0.6884, 0.150),
    'malachite_ore': (-0.3352, 0.6884, 0.150),
    'vanadinite_ore': (-0.6266, 0.6884, 0.150),
}

# Height Waypoints (m in base_link)
SAFE_HOME_POSE = (-0.5000, -0.0500, 0.3500)   # Neutral intermediate transit waypoint
PRE_PICK_HEIGHT_Z = 0.1500                    # Safe height above conveyor ore before descent
LIFT_HEIGHT_Z = 0.3000                        # Safe transit height after picking ore

# Container rim height from Kepler Colony SDF (model.sdf cargo_rover_link z=0.5688 + mesh rim y=0.2814 - mount_z=0.8825)
CONTAINER_RIM_Z = -0.0323                     # Measured container rim height in base_link (m)
DROP_DESCENT_HEIGHT_Z = CONTAINER_RIM_Z + 0.0762 + 0.03   # 0.0739 m: Descent height into container for clean release
PRE_DROP_HEIGHT_Z = DROP_DESCENT_HEIGHT_Z + 0.06          # 0.1339 m: Approach height above cargo container
COMFORT_REACH_M = 0.82              # Reach below which no tilt is needed (m)
TOOL_LEN_M = 0.25                   # Effective tool length for tilt calculation (m)
MAX_DROP_TILT_DEG = 35              # Maximum allowed tool tilt toward container (deg)
MAX_DROP_REACH_M = 0.85              # Max XY reach for straight-line container approach (m)
DROP_MIN_Y_M = 0.575                 # Min y for clamped drop target (~3 cm clear of bay front wall at y=0.4884)
FAR_DROP_PRE_Z = DROP_DESCENT_HEIGHT_Z + 0.02  # ~0.094 m: lower approach for far containers
FAR_DROP_SPEED_MPS = 0.08            # Slower cruise for far container approach (m/s)

# Motion Speeds & Acceleration Limits
CRUISE_SPEED_MPS = 0.12                       # Main transit linear velocity (m/s)
MIN_SERVO_SPEED_MPS = 0.008                   # Minimum speed floor during Cartesian moves
PICK_DESCENT_SPEED_MPS = 0.025                # Slow descent to ore to avoid impact (m/s)
DROP_DESCENT_SPEED_MPS = 0.050                # Gentle descent into container (m/s)
POST_CONTACT_LIFT_SPEED_MPS = 0.050           # Lift speed immediately after contact (m/s)
CAP_LINEAR_MPS = 0.15                         # Safety velocity cap (m/s)
CAP_ANGULAR_RPS = 0.35                        # Safety angular velocity cap (rad/s)
LINEAR_ACCEL_LIMIT = 0.40                     # Max linear acceleration (m/s^2)
ANGULAR_ACCEL_LIMIT = 0.80                    # Max angular acceleration (rad/s^2)

# Control Gains & Tolerances
KP_POSITION = 1.30                            # Proportional gain for Cartesian position error
KP_ORIENTATION = 1.00                         # Proportional gain for vertical tool orientation hold
KP_JOINT = 1.00                               # Proportional gain for initial joint unfolding
POSITION_TOLERANCE_M = 0.012                  # General position arrival tolerance (m)
PRE_PICK_TOL_M = 0.004                        # Pre-pick arrival tolerance (m)
DESCENT_TOL_M = 0.003                         # Fine descent arrival tolerance (m)
STANDSTILL_HOLD_S = 0.50                      # Settle pause at waypoints / post contact (s)
FINE_SETTLE_S = 0.6          # hold still at fine-approach height so tool spring settles
CONTACT_CONFIRM_S = 0.10     # force must stay >= CONTACT_FORCE_TRIGGER_N this long
FINE_APPROACH_M = 0.030                       # Height above ore to switch to fine descent (m)
PICK_OVERSHOOT_M = 0.005                      # Target overshoot depth (m)
POST_CONTACT_LIFT_M = 0.050                   # Small lift distance to test captured ore (m)
POST_CONTACT_WAIT_S = 0.30                    # Settle pause before verifying grip (s)
DESCENT_LOG_INTERVAL_S = 0.25                 # Log interval for descent telemetry (s)

# Force Sensor Thresholds (/ur7e/end_effector/force_magnitude in N)
CONTACT_FORCE_TRIGGER_N = 1.0                 # Stop descent when tool touches ore surface (N)
FORCE_PROTECTIVE_LIMIT_N = 4.0                # Safety back-off threshold if contact is too hard (N)
HELD_CONFIRM_FORCE_N = 0.70                   # Force indicating ore weight is supported on tool (N)
RELEASE_CONFIRM_FORCE_N = 0.40                # Force indicating ore has detached (N)
RELEASE_CONFIRM_WINDOW_S = 0.30               # Duration force must remain low to confirm release (s)

# Discovery & Settle Parameters
ORE_Z_MIN_M = -0.085
ORE_Z_MAX_M = -0.045
DISCOVER_SETTLE_S = 2.0                       # Time window required for stable frames (s)
DISCOVER_TIMEOUT_S = 15.0                     # Max timeout for ore discovery before proceeding (s)
DISCOVER_MAX_TF_AGE_S = 1.0                   # Max acceptable age of TF transform (s)
DISCOVER_MAX_SPREAD_M = 0.003                 # Max per-axis coordinate spread (3 mm)
DISCOVER_REQUIRED_COUNT = 4                   # Number of distinct stable frames to wait for

# Retry & Timeout Limits
MAX_LATCHES_BEFORE_RETREAT = 2               # Max consecutive latches before joint retreat
MAX_HELD_RETREATS_PER_ORE = 4                 # Max retreats for held ore before stopping
MAX_PICK_ATTEMPTS = 3                         # Maximum retries per ore before queueing second pass
MAX_RELEASE_RETRIES = 3                       # Maximum re-sends of magnet OFF
MAGNET_RETRY_INTERVAL_S = 0.50                # Delay between service call retries (s)
MAGNET_MAX_RETRIES = 10                       # Max service call attempts
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

def quat_rotate(q, v):
    vx, vy, vz = v
    x, y, z, w = q
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def shortest_arc_quat(v_from, v_to):
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(v_from, v_to))))
    if dot > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -0.999999:
        axis = (1.0, 0.0, 0.0) if abs(v_from[0]) < 0.9 else (0.0, 1.0, 0.0)
        cross = (
            axis[1] * v_from[2] - axis[2] * v_from[1],
            axis[2] * v_from[0] - axis[0] * v_from[2],
            axis[0] * v_from[1] - axis[1] * v_from[0],
        )
        n = math.sqrt(sum(c * c for c in cross))
        return (cross[0] / n, cross[1] / n, cross[2] / n, 0.0)
    cross = (
        v_from[1] * v_to[2] - v_from[2] * v_to[1],
        v_from[2] * v_to[0] - v_from[0] * v_to[2],
        v_from[0] * v_to[1] - v_from[1] * v_to[0],
    )
    w = 1.0 + dot
    return quat_normalize((cross[0], cross[1], cross[2], w))


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
    STATE_ENABLE_MAGNET = 7
    STATE_DESCENT_PICK = 8
    STATE_DESCENT_BACKOFF = 9
    STATE_CONTACT_HOLD = 10
    STATE_CONTACT_LIFT = 11
    STATE_CONFIRM_GRIP = 12
    STATE_LIFT_AFTER_PICK = 13
    STATE_TRANSIT_HOME_WITH_ORE = 14
    STATE_APPROACH_PRE_DROP = 15
    STATE_DESCENT_DROP = 16
    STATE_DISABLE_MAGNET = 17
    STATE_CONFIRM_RELEASE = 18
    STATE_RETREAT_TO_PRE_DROP = 19
    STATE_RETURN_HOME = 20
    STATE_LATCH_RECOVERY_UNLOCK = 21
    STATE_LATCH_RECOVERY_CLEAR = 22
    STATE_ALL_COMPLETED = 23

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

        # Lifecycle & Ore Tracking
        self.discovered = False
        self.ore_held = False
        self.fine_stage = False
        self.fine_enter_time = None
        self.drop_xy_logged = {}  # cache: kind -> (x, y), logs only once per kind
        self.vertical_orientation = None  # tool-down quaternion set in STATE_SWITCH_TO_TWIST
        self.contact_start = None
        self.latch_count = 0
        self.retreat_resume_state = None
        self.ore_retreat_counts = {}
        self.done_ores = set()
        self.second_pass_ores = set()
        self.logged_tcp_frame = False

        # Discovery tracking
        self.discovery_start_time = None
        self.discovery_history = {}  # fid -> [(t_stamp, tx, ty, tz)]

        # Telemetry & Retry Counters
        self.magnet_retry_count = 0
        self.last_magnet_call_time = 0.0
        self.release_retry_count = 0
        self.release_force_ok_start_time = None
        self.last_descent_log_time = 0.0
        self.recent_forces = []  # [(t, force)]
        self.contact_z = 0.0

        # Timing variables
        self.state_enter_time = 0.0
        self.last_log_time = 0.0
        self.node_init_time = None

        # Main Control Timer (20 Hz)
        self.timer = self.create_timer(CONTROL_PERIOD_S, self.step_state_machine)
        self.get_logger().info('Task 2A Manipulation Node initialized.')
        self.get_logger().info(
            f'Cargo Container Heights: rim_z={CONTAINER_RIM_Z:.4f}m, '
            f'DROP_DESCENT_HEIGHT_Z={DROP_DESCENT_HEIGHT_Z:.4f}m, '
            f'PRE_DROP_HEIGHT_Z={PRE_DROP_HEIGHT_Z:.4f}m'
        )

    def tcpposecb(self, data):
        self.tcp_pose = data.pose
        if not self.logged_tcp_frame:
            self.get_logger().info(f'/tcp_pose_raw header.frame_id: {data.header.frame_id}')
            self.logged_tcp_frame = True

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
        now = self.get_clock().now().nanoseconds / 1e9
        self.recent_forces.append((now, self.force_magnitude))
        self.recent_forces = [f for f in self.recent_forces if now - f[0] <= 1.0]

    def change_state(self, new_state, now):
        old_state = self.state
        latch_states = (self.STATE_LATCH_RECOVERY_UNLOCK, self.STATE_LATCH_RECOVERY_CLEAR)
        if new_state != old_state and old_state not in latch_states and new_state not in latch_states:
            self.latch_count = 0

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

    def tilted_orientation(self, x, y):
        r = math.hypot(x, y)
        if r < 1e-6 or self.vertical_orientation is None:
            return self.vertical_orientation
        excess = max(0.0, r - COMFORT_REACH_M)
        tilt = min(max(0.0, math.asin(min(1.0, excess / TOOL_LEN_M))),
                   math.radians(MAX_DROP_TILT_DEG))
        if tilt < 1e-4:
            return self.vertical_orientation
        ux, uy = x / r, y / r
        d = (ux * math.sin(tilt), uy * math.sin(tilt), -math.cos(tilt))
        q_delta = shortest_arc_quat((0.0, 0.0, -1.0), d)
        return quat_normalize(quat_multiply(q_delta, self.vertical_orientation))

    def drop_xy(self, kind):
        if kind in self.drop_xy_logged:
            return self.drop_xy_logged[kind]
        cx, cy, _ = CONTAINER_DROP_POSES[kind]
        dist = math.hypot(cx, cy)
        self.get_logger().info(f'[DROP-XY] {kind}: centre ({cx:.4f},{cy:.4f}) dist={dist:.4f}m')
        result = (cx, cy)
        self.drop_xy_logged[kind] = result
        return result

    def navigate_cartesian(self, target_xyz, cruise_speed, now, tol=POSITION_TOLERANCE_M):
        '''
        Servo the TCP toward target_xyz in base_link with orientation stabilization.
        Returns True when distance <= tol.
        '''
        cx, cy, cz = self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z
        tx, ty, tz = target_xyz
        error_vec = (tx - cx, ty - cy, tz - cz)
        dist = vector_norm(error_vec)

        if dist <= tol:
            self.publish_zero_twist()
            return True

        unit_dir = tuple(e / dist for e in error_vec)
        speed = max(MIN_SERVO_SPEED_MPS, min(cruise_speed, KP_POSITION * dist))
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
                self.latch_count += 1
                tcp_str = f'({self.tcp_pose.position.x:.4f}, {self.tcp_pose.position.y:.4f}, {self.tcp_pose.position.z:.4f})' if self.tcp_pose else 'None'
                joints_str = ', '.join(f'{k}: {self.joint_angles.get(k, 0.0):.3f}' for k in JOINT_NAMES)
                ore_name = self.current_ore[0] if self.current_ore else 'None'
                target_xyz = f'({self.current_ore[2][0]:.4f}, {self.current_ore[2][1]:.4f}, {self.current_ore[2][2]:.4f})' if self.current_ore else 'None'
                self.get_logger().error(
                    f'PROTECTIVE STOP DETECTED: code={self.arm_status} (latch_count={self.latch_count}). '
                    f'State: {self.state_name(self.state)}, TCP: {tcp_str}, Joints: [{joints_str}], '
                    f'Ore: {ore_name}, Target: {target_xyz}. Latching position and initiating recovery...'
                )
                self.get_logger().info(
                    f'[DROP-JOINTS] Ore: {self.current_ore[1] if self.current_ore else "None"}, TCP: {tcp_str}, Joints: [{joints_str}]'
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
                        self.get_logger().info('Switched to twist controller. Aligning tool orientation...')
                        current_q = quat_normalize((
                            self.tcp_pose.orientation.x,
                            self.tcp_pose.orientation.y,
                            self.tcp_pose.orientation.z,
                            self.tcp_pose.orientation.w,
                        ))
                        # Compute tool z-axis (+z out of magnet face) in base_link
                        tool_z = quat_rotate(current_q, (0.0, 0.0, 1.0))
                        dot = max(-1.0, min(1.0, -tool_z[2]))
                        angle_deg = math.degrees(math.acos(dot))
                        self.get_logger().info(f'Tool z-axis angle from (0,0,-1): {angle_deg:.2f} deg')
                        if angle_deg > 2.0:
                            q_delta = shortest_arc_quat(tool_z, (0.0, 0.0, -1.0))
                            self.target_orientation = quat_normalize(quat_multiply(q_delta, current_q))
                            self.get_logger().info('Built target_orientation via shortest-arc rotation to tool z = (0,0,-1).')
                        else:
                            self.target_orientation = current_q
                        self.vertical_orientation = self.target_orientation  # store tool-down reference

                        self.previous_linear = (0.0, 0.0, 0.0)
                        self.previous_angular = (0.0, 0.0, 0.0)
                        if self.retreat_resume_state is not None:
                            resume = self.retreat_resume_state
                            self.retreat_resume_state = None
                            self.change_state(resume, now)
                        else:
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
                if self.discovered:
                    self.ore_queue = [o for o in self.ore_queue if o[0] not in self.done_ores]
                    if len(self.ore_queue) > 0:
                        self.current_ore = self.ore_queue.pop(0)
                        self.pick_attempt_count = 0
                        self.change_state(self.STATE_APPROACH_PRE_PICK, now)
                    else:
                        self.get_logger().info('All completed!')
                        self.change_state(self.STATE_ALL_COMPLETED, now)
                else:
                    self.change_state(self.STATE_DISCOVER_ORES, now)
            return

        # ------------------------------------------------------------------
        # STATE 5: Discover Available Ores via TF Buffer
        # ------------------------------------------------------------------
        elif self.state == self.STATE_DISCOVER_ORES:
            if self.discovery_start_time is None:
                self.discovery_start_time = now

            node_time_sec = self.get_clock().now().nanoseconds / 1e9
            candidate_kinds = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

            for kind in candidate_kinds:
                for idx in range(1, 5):
                    frame_id = f'{kind}_{idx}'
                    try:
                        tf = self.tf_buffer.lookup_transform(BASE_FRAME, frame_id, Time())
                        tf_sec = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                        age = node_time_sec - tf_sec
                        if 0.0 <= age < DISCOVER_MAX_TF_AGE_S:
                            tx = tf.transform.translation.x
                            ty = tf.transform.translation.y
                            tz = tf.transform.translation.z
                            if frame_id not in self.discovery_history:
                                self.discovery_history[frame_id] = []
                            self.discovery_history[frame_id].append((now, tx, ty, tz))
                    except Exception:
                        pass

            stable_ores = {}
            for fid, samples in self.discovery_history.items():
                recent = [s for s in samples if now - s[0] <= DISCOVER_SETTLE_S]
                self.discovery_history[fid] = recent
                if len(recent) >= 5 and (recent[-1][0] - recent[0][0]) >= (DISCOVER_SETTLE_S - 0.2):
                    if (now - recent[-1][0]) < DISCOVER_MAX_TF_AGE_S:
                        xs = [s[1] for s in recent]
                        ys = [s[2] for s in recent]
                        zs = [s[3] for s in recent]
                        if (max(xs) - min(xs) < DISCOVER_MAX_SPREAD_M and
                            max(ys) - min(ys) < DISCOVER_MAX_SPREAD_M and
                            max(zs) - min(zs) < DISCOVER_MAX_SPREAD_M):
                            med_x = float(statistics.median(xs))
                            med_y = float(statistics.median(ys))
                            med_z = float(statistics.median(zs))
                            kind = fid.rsplit('_', 1)[0]
                            if ORE_Z_MIN_M <= med_z <= ORE_Z_MAX_M:
                                stable_ores[fid] = (fid, kind, (med_x, med_y, med_z))

            timeout_expired = (now - self.discovery_start_time) >= DISCOVER_TIMEOUT_S
            if len(stable_ores) >= DISCOVER_REQUIRED_COUNT or (timeout_expired and len(stable_ores) >= 1):
                ore_list = list(stable_ores.values())
                # Order by xy distance from base_link origin, nearest first
                ore_list.sort(key=lambda item: math.hypot(item[2][0], item[2][1]))
                self.ore_queue = ore_list
                self.discovered = True
                self.get_logger().info(f'Discovered {len(self.ore_queue)} ores: {[o[0] for o in self.ore_queue]}')
                self.current_ore = self.ore_queue.pop(0)
                self.pick_attempt_count = 0
                self.change_state(self.STATE_APPROACH_PRE_PICK, now)
            else:
                if now - self.last_log_time > 1.0:
                    self.get_logger().info(
                        f'Sampling ores: {len(stable_ores)} stable, time={now - self.discovery_start_time:.1f}s'
                    )
                    self.last_log_time = now
            return

        # ------------------------------------------------------------------
        # STATE 6: Move to Pre-Pick Pose (Above Ore)
        # ------------------------------------------------------------------
        elif self.state == self.STATE_APPROACH_PRE_PICK:
            if self.vertical_orientation is not None:
                self.target_orientation = self.vertical_orientation
            ox, oy, _ = self.current_ore[2]
            target_pre_pick = (ox, oy, PRE_PICK_HEIGHT_Z)
            arrived = self.navigate_cartesian(target_pre_pick, CRUISE_SPEED_MPS, now, tol=PRE_PICK_TOL_M)
            if arrived:
                self.fine_stage = False
                self.fine_enter_time = None
                self.contact_start = None
                self.magnet_retry_count = 0
                self.last_magnet_call_time = 0.0
                self.magnet_future = None
                self.change_state(self.STATE_ENABLE_MAGNET, now)
            return

        # ------------------------------------------------------------------
        # STATE 7: Enable Electromagnet on Pre-Pick Arrival
        # ------------------------------------------------------------------
        elif self.state == self.STATE_ENABLE_MAGNET:
            if self.magnet_future is None:
                if (now - self.last_magnet_call_time) >= MAGNET_RETRY_INTERVAL_S:
                    if self.magnet_cli.service_is_ready():
                        self.call_magnet(True)
                    self.magnet_retry_count += 1
                    self.last_magnet_call_time = now
                    if self.magnet_retry_count > MAGNET_MAX_RETRIES:
                        self.get_logger().error('Magnet ON failed after max retries; proceeding to descent.')
                        self.change_state(self.STATE_DESCENT_PICK, now)
            else:
                if self.magnet_future.done():
                    try:
                        res = self.magnet_future.result()
                        if res.success:
                            self.magnet_future = None
                            self.last_descent_log_time = 0.0
                            self.change_state(self.STATE_DESCENT_PICK, now)
                            return
                        else:
                            self.magnet_future = None
                    except Exception as e:
                        self.get_logger().error(f'Magnet service error: {e}')
                        self.magnet_future = None
            return

        # ------------------------------------------------------------------
        # STATE 8: Two-Stage Descent to Ore with Force Monitoring
        # ------------------------------------------------------------------
        elif self.state == self.STATE_DESCENT_PICK:
            ox, oy, oz = self.current_ore[2]
            cz = self.tcp_pose.position.z

            if (now - self.last_descent_log_time) >= DESCENT_LOG_INTERVAL_S:
                self.get_logger().info(f'Descent telemetry: z={cz:.4f}m, oz={oz:.4f}m, force={self.force_magnitude:.2f}N')
                self.last_descent_log_time = now

            if self.force_magnitude >= FORCE_PROTECTIVE_LIMIT_N:
                self.publish_zero_twist()
                self.get_logger().warn(
                    f'Force exceeded protective limit ({self.force_magnitude:.2f}N >= {FORCE_PROTECTIVE_LIMIT_N}N). '
                    f'Moving up 2 cm and counting failed attempt.'
                )
                self.pick_attempt_count += 1
                self.contact_z = cz
                self.change_state(self.STATE_DESCENT_BACKOFF, now)
                return

            armed = (self.fine_stage and self.fine_enter_time is not None
                     and (now - self.fine_enter_time) >= FINE_SETTLE_S)

            if armed and self.force_magnitude >= CONTACT_FORCE_TRIGGER_N:
                if self.contact_start is None:
                    self.contact_start = now
                elif (now - self.contact_start) >= CONTACT_CONFIRM_S:
                    self.publish_zero_twist()
                    self.contact_z = cz
                    self.get_logger().info(f'Contact detected on {self.current_ore[0]}: force={self.force_magnitude:.2f}N at z={cz:.4f}m (ore top z={oz:.4f})')
                    self.change_state(self.STATE_CONTACT_HOLD, now)
                    return
            else:
                self.contact_start = None

            if not self.fine_stage:
                if self.navigate_cartesian((ox, oy, oz + FINE_APPROACH_M), CRUISE_SPEED_MPS, now, tol=DESCENT_TOL_M):
                    self.fine_stage = True
                    self.fine_enter_time = now
                    self.publish_zero_twist()
            elif not armed:
                self.publish_zero_twist()
            else:
                arrived = self.navigate_cartesian((ox, oy, oz - PICK_OVERSHOOT_M), PICK_DESCENT_SPEED_MPS, now, tol=DESCENT_TOL_M)
                if arrived:
                    self.publish_zero_twist()
                    self.contact_z = cz
                    self.change_state(self.STATE_CONTACT_HOLD, now)
            return

        # ------------------------------------------------------------------
        # STATE 9: Descent Safety Backoff (move up 2 cm)
        # ------------------------------------------------------------------
        elif self.state == self.STATE_DESCENT_BACKOFF:
            ox, oy, _ = self.current_ore[2]
            target_backoff = (ox, oy, self.contact_z + 0.02)
            arrived = self.navigate_cartesian(target_backoff, PICK_DESCENT_SPEED_MPS, now, tol=POSITION_TOLERANCE_M)
            if arrived:
                self.call_magnet(False)
                if self.pick_attempt_count < MAX_PICK_ATTEMPTS:
                    self.change_state(self.STATE_APPROACH_PRE_PICK, now)
                else:
                    self.get_logger().warn(f'Max pick attempts exceeded for {self.current_ore[0]}.')
                    if self.current_ore[0] not in self.second_pass_ores:
                        self.second_pass_ores.add(self.current_ore[0])
                        self.ore_queue.append(self.current_ore)
                    self.ore_queue = [o for o in self.ore_queue if o[0] not in self.done_ores]
                    if len(self.ore_queue) > 0:
                        self.current_ore = self.ore_queue.pop(0)
                        self.pick_attempt_count = 0
                        self.change_state(self.STATE_APPROACH_PRE_PICK, now)
                    else:
                        self.change_state(self.STATE_ALL_COMPLETED, now)
            return

        # ------------------------------------------------------------------
        # STATE 10: Hold Still Post Contact
        # ------------------------------------------------------------------
        elif self.state == self.STATE_CONTACT_HOLD:
            if (now - self.state_enter_time) >= STANDSTILL_HOLD_S:
                self.change_state(self.STATE_CONTACT_LIFT, now)
            return

        # ------------------------------------------------------------------
        # STATE 11: Small Test Lift Post Contact (0.05m at 0.05 m/s)
        # ------------------------------------------------------------------
        elif self.state == self.STATE_CONTACT_LIFT:
            ox, oy, _ = self.current_ore[2]
            target_test_lift = (ox, oy, self.contact_z + POST_CONTACT_LIFT_M)
            arrived = self.navigate_cartesian(target_test_lift, POST_CONTACT_LIFT_SPEED_MPS, now, tol=POSITION_TOLERANCE_M)
            if arrived:
                self.publish_zero_twist()
                self.change_state(self.STATE_CONFIRM_GRIP, now)
            return

        # ------------------------------------------------------------------
        # STATE 12: Wait 0.3s & Confirm Grip via Mean Force
        # ------------------------------------------------------------------
        elif self.state == self.STATE_CONFIRM_GRIP:
            if (now - self.state_enter_time) >= POST_CONTACT_WAIT_S:
                forces_in_window = [f[1] for f in self.recent_forces if (now - f[0]) <= POST_CONTACT_WAIT_S]
                mean_force = sum(forces_in_window) / len(forces_in_window) if forces_in_window else self.force_magnitude
                if mean_force >= HELD_CONFIRM_FORCE_N:
                    self.get_logger().info(
                        f'SUCCESSFUL PICK: {self.current_ore[0]} confirmed held (mean force={mean_force:.2f}N).'
                    )
                    picked_msg = String()
                    picked_msg.data = self.current_ore[0]
                    self.picked_pub.publish(picked_msg)
                    self.ore_held = True
                    self.change_state(self.STATE_LIFT_AFTER_PICK, now)
                else:
                    self.pick_attempt_count += 1
                    self.get_logger().warn(
                        f'Pick failed for {self.current_ore[0]} (mean force={mean_force:.2f}N < {HELD_CONFIRM_FORCE_N}N). '
                        f'Attempt {self.pick_attempt_count}/{MAX_PICK_ATTEMPTS}'
                    )
                    self.call_magnet(False)
                    if self.pick_attempt_count < MAX_PICK_ATTEMPTS:
                        self.change_state(self.STATE_APPROACH_PRE_PICK, now)
                    else:
                        self.get_logger().warn(
                            f'Max pick attempts exceeded for {self.current_ore[0]}. Appending to queue end for second pass.'
                        )
                        if self.current_ore[0] not in self.second_pass_ores:
                            self.second_pass_ores.add(self.current_ore[0])
                            self.ore_queue.append(self.current_ore)
                        self.ore_queue = [o for o in self.ore_queue if o[0] not in self.done_ores]
                        if len(self.ore_queue) > 0:
                            self.current_ore = self.ore_queue.pop(0)
                            self.pick_attempt_count = 0
                            self.change_state(self.STATE_APPROACH_PRE_PICK, now)
                        else:
                            self.change_state(self.STATE_ALL_COMPLETED, now)
            return

        # ------------------------------------------------------------------
        # STATE 13: Full Lift to Transit Height
        # ------------------------------------------------------------------
        elif self.state == self.STATE_LIFT_AFTER_PICK:
            ox, oy, _ = self.current_ore[2]
            target_lift = (ox, oy, LIFT_HEIGHT_Z)
            arrived = self.navigate_cartesian(target_lift, CRUISE_SPEED_MPS, now, tol=POSITION_TOLERANCE_M)
            if arrived:
                self.change_state(self.STATE_TRANSIT_HOME_WITH_ORE, now)
            return

        # ------------------------------------------------------------------
        # STATE 14: Transit Home Waypoint with Ore
        # ------------------------------------------------------------------
        elif self.state == self.STATE_TRANSIT_HOME_WITH_ORE:
            arrived = self.navigate_cartesian(SAFE_HOME_POSE, CRUISE_SPEED_MPS, now)
            if arrived:
                self.change_state(self.STATE_APPROACH_PRE_DROP, now)
            return

        # ------------------------------------------------------------------
        # STATE 15: Move to Pre-Drop Pose Above Container
        # ------------------------------------------------------------------
        elif self.state == self.STATE_APPROACH_PRE_DROP:
            kind = self.current_ore[1]
            cx_d, cy_d, _ = CONTAINER_DROP_POSES[kind]
            self.target_orientation = self.tilted_orientation(cx_d, cy_d)
            drop_x, drop_y = self.drop_xy(kind)
            is_far = math.hypot(drop_x, drop_y) > MAX_DROP_REACH_M
            pre_z = FAR_DROP_PRE_Z if is_far else PRE_DROP_HEIGHT_Z
            speed = FAR_DROP_SPEED_MPS if is_far else CRUISE_SPEED_MPS
            arrived = self.navigate_cartesian((drop_x, drop_y, pre_z), speed, now)
            if arrived:
                self.change_state(self.STATE_DESCENT_DROP, now)
            return

        # ------------------------------------------------------------------
        # STATE 16: Descend into Container
        # ------------------------------------------------------------------
        elif self.state == self.STATE_DESCENT_DROP:
            kind = self.current_ore[1]
            cx_d, cy_d, _ = CONTAINER_DROP_POSES[kind]
            self.target_orientation = self.tilted_orientation(cx_d, cy_d)
            drop_x, drop_y = self.drop_xy(kind)
            target_drop = (drop_x, drop_y, DROP_DESCENT_HEIGHT_Z)
            arrived = self.navigate_cartesian(target_drop, DROP_DESCENT_SPEED_MPS, now)
            if arrived:
                self.publish_zero_twist()
                self.magnet_retry_count = 0
                self.last_magnet_call_time = 0.0
                self.magnet_future = None
                self.release_retry_count = 0
                self.release_force_ok_start_time = None
                self.change_state(self.STATE_DISABLE_MAGNET, now)
            return

        # ------------------------------------------------------------------
        # STATE 17: De-energize Magnet with Retry Logic
        # ------------------------------------------------------------------
        elif self.state == self.STATE_DISABLE_MAGNET:
            if self.magnet_future is None:
                if (now - self.last_magnet_call_time) >= MAGNET_RETRY_INTERVAL_S:
                    if self.magnet_cli.service_is_ready():
                        self.call_magnet(False)
                    self.magnet_retry_count += 1
                    self.last_magnet_call_time = now
                    if self.magnet_retry_count > MAGNET_MAX_RETRIES:
                        self.get_logger().error('Magnet OFF call failed max retries; proceeding to release confirm.')
                        self.change_state(self.STATE_CONFIRM_RELEASE, now)
            else:
                if self.magnet_future.done():
                    try:
                        res = self.magnet_future.result()
                        if res.success:
                            self.magnet_future = None
                            self.release_force_ok_start_time = None
                            self.change_state(self.STATE_CONFIRM_RELEASE, now)
                            return
                        else:
                            self.magnet_future = None
                    except Exception as e:
                        self.get_logger().error(f'Magnet OFF error: {e}')
                        self.magnet_future = None
            return

        # ------------------------------------------------------------------
        # STATE 18: Confirm Release via Force Threshold Check
        # ------------------------------------------------------------------
        elif self.state == self.STATE_CONFIRM_RELEASE:
            if self.force_magnitude < RELEASE_CONFIRM_FORCE_N:
                if self.release_force_ok_start_time is None:
                    self.release_force_ok_start_time = now
                elif (now - self.release_force_ok_start_time) >= RELEASE_CONFIRM_WINDOW_S:
                    tcp_str = f'({self.tcp_pose.position.x:.4f}, {self.tcp_pose.position.y:.4f}, {self.tcp_pose.position.z:.4f})' if self.tcp_pose else 'None'
                    joints_str = ', '.join(f'{k}: {self.joint_angles.get(k, 0.0):.3f}' for k in JOINT_NAMES)
                    cx_d, cy_d, _ = CONTAINER_DROP_POSES[self.current_ore[1]]
                    tilt_deg = math.degrees(min(max(0.0, math.asin(min(1.0, max(0.0, math.hypot(cx_d, cy_d) - COMFORT_REACH_M) / TOOL_LEN_M))), math.radians(MAX_DROP_TILT_DEG)))
                    self.get_logger().info(
                        f'SUCCESSFUL DROP: {self.current_ore[0]} released into container (force={self.force_magnitude:.2f}N).'
                    )
                    self.get_logger().info(
                        f'[DROP-JOINTS] Ore: {self.current_ore[1]}, tilt={tilt_deg:.2f}deg, TCP: {tcp_str}, Joints: [{joints_str}]'
                    )
                    self.done_ores.add(self.current_ore[0])
                    self.ore_held = False
                    self.change_state(self.STATE_RETREAT_TO_PRE_DROP, now)
                    return
            else:
                self.release_force_ok_start_time = None
                if (now - self.state_enter_time) >= 0.5:
                    if self.release_retry_count < MAX_RELEASE_RETRIES:
                        self.release_retry_count += 1
                        self.get_logger().warn(
                            f'Release force {self.force_magnitude:.2f}N >= {RELEASE_CONFIRM_FORCE_N}N. '
                            f'Re-sending magnet OFF ({self.release_retry_count}/{MAX_RELEASE_RETRIES})'
                        )
                        self.call_magnet(False)
                        self.state_enter_time = now
                    else:
                        self.get_logger().warn('Max release retries reached; concluding drop.')
                        self.done_ores.add(self.current_ore[0])
                        self.ore_held = False
                        self.change_state(self.STATE_RETREAT_TO_PRE_DROP, now)
                        return
            return

        # ------------------------------------------------------------------
        # STATE 19: Retreat Upward to Pre-Drop Height
        # ------------------------------------------------------------------
        elif self.state == self.STATE_RETREAT_TO_PRE_DROP:
            kind = self.current_ore[1]
            cx_d, cy_d, _ = CONTAINER_DROP_POSES[kind]
            self.target_orientation = self.tilted_orientation(cx_d, cy_d)
            drop_x, drop_y = self.drop_xy(kind)
            is_far = math.hypot(drop_x, drop_y) > MAX_DROP_REACH_M
            if is_far:
                arrived = self.navigate_cartesian(SAFE_HOME_POSE, CRUISE_SPEED_MPS, now)
                if arrived:
                    self.target_orientation = self.vertical_orientation
                    self.change_state(self.STATE_RETURN_HOME, now)
            else:
                pre_z = PRE_DROP_HEIGHT_Z
                arrived = self.navigate_cartesian((drop_x, drop_y, pre_z), CRUISE_SPEED_MPS, now)
                if arrived:
                    self.target_orientation = self.vertical_orientation
                    self.change_state(self.STATE_RETURN_HOME, now)
            return

        # ------------------------------------------------------------------
        # STATE 20: Return to Safe Home Waypoint & Check Remaining Ores
        # ------------------------------------------------------------------
        elif self.state == self.STATE_RETURN_HOME:
            arrived = self.navigate_cartesian(SAFE_HOME_POSE, CRUISE_SPEED_MPS, now)
            if arrived:
                self.ore_queue = [o for o in self.ore_queue if o[0] not in self.done_ores]
                if len(self.ore_queue) > 0:
                    self.current_ore = self.ore_queue.pop(0)
                    self.pick_attempt_count = 0
                    self.change_state(self.STATE_APPROACH_PRE_PICK, now)
                else:
                    self.get_logger().info('All completed!')
                    self.change_state(self.STATE_ALL_COMPLETED, now)
            return

        # ------------------------------------------------------------------
        # STATE 21: Latch Recovery -- Unlock Service Call
        # ------------------------------------------------------------------
        elif self.state == self.STATE_LATCH_RECOVERY_UNLOCK:
            if self.unlock_future is None:
                self.call_unlock()
            else:
                if self.unlock_future.done():
                    try:
                        res = self.unlock_future.result()
                        self.get_logger().info(f'Unlock protective stop returned: {res.message}')
                    except Exception as e:
                        self.get_logger().error(f'Unlock service error: {e}')
                    self.unlock_future = None
                    self.change_state(self.STATE_LATCH_RECOVERY_CLEAR, now)
            return

        # ------------------------------------------------------------------
        # STATE 22: Latch Recovery -- Immediate Upward Motion (Clear Fault)
        # ------------------------------------------------------------------
        elif self.state == self.STATE_LATCH_RECOVERY_CLEAR:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = BASE_FRAME
            msg.twist.linear.z = 0.05
            self.twist_pub.publish(msg)

            if self.arm_status == 0 or (now - self.state_enter_time) >= 1.5:
                self.publish_zero_twist()
                if self.latch_count >= MAX_LATCHES_BEFORE_RETREAT:
                    if self.ore_held:
                        ore_id = self.current_ore[0] if self.current_ore else 'unknown'
                        self.ore_retreat_counts[ore_id] = self.ore_retreat_counts.get(ore_id, 0) + 1
                        self.get_logger().warn(
                            f'Held-ore retreat count for {ore_id}: {self.ore_retreat_counts[ore_id]}/{MAX_HELD_RETREATS_PER_ORE}'
                        )
                        if self.ore_retreat_counts[ore_id] >= MAX_HELD_RETREATS_PER_ORE:
                            self.get_logger().error(
                                f'Ore {ore_id} caused {self.ore_retreat_counts[ore_id]} joint retreats while held. '
                                f'Stopping run (STATE_ALL_COMPLETED) to prevent penalties.'
                            )
                            self.publish_zero_twist()
                            self.change_state(self.STATE_ALL_COMPLETED, now)
                            return

                    self.get_logger().warn(
                        f'Latch count {self.latch_count} >= {MAX_LATCHES_BEFORE_RETREAT}. Initiating joint-space retreat to unfold targets...'
                    )
                    if self.ore_held is False and (self.current_ore is None or self.current_ore[0] not in self.done_ores):
                        self.call_magnet(False)
                        if self.current_ore is not None:
                            self.pick_attempt_count += 1
                            self.ore_queue.insert(0, self.current_ore)
                            self.current_ore = None
                    self.retreat_resume_state = self.STATE_TRANSIT_HOME_WITH_ORE if self.ore_held else self.STATE_MOVE_HOME
                    self.latch_count = 0
                    self.change_state(self.STATE_SWITCH_TO_JOINT, now)
                    self.switch_controller(JOINT_CONTROLLER)
                    return

                self.get_logger().info('Arm fault cleared. Resuming appropriate workflow...')
                if self.ore_held is False and (self.current_ore is None or self.current_ore[0] not in self.done_ores):
                    self.call_magnet(False)
                    if self.current_ore is not None:
                        self.pick_attempt_count += 1
                        self.ore_queue.insert(0, self.current_ore)
                        self.current_ore = None
                    if self.vertical_orientation is not None:
                        self.target_orientation = self.vertical_orientation
                    self.change_state(self.STATE_MOVE_HOME, now)
                elif self.ore_held is True:
                    self.change_state(self.STATE_TRANSIT_HOME_WITH_ORE, now)
                else:
                    self.change_state(self.STATE_RETURN_HOME, now)
            return

        # ------------------------------------------------------------------
        # STATE 23: All Tasks Completed
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
