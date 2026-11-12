#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 3A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task3A_manipulation.py
# Functions:
#                   quat_normalize, quat_conjugate, quat_multiply, quaternion_error,
#                   vector_norm, cap_vector, rate_limit_vector, quat_rotate,
#                   shortest_arc_quat, main
# Nodes:            Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds, /ore_picked,
#                                          /package_lifted ]
#                   Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status,
#                                          /ur7e/end_effector/force_magnitude, /tf, /ebot_at_arm ]
#                   Service Clients    - [ /ur_arm_controller/switch_controller, /magnet,
#                                          /ur_arm_controller/unlock_protective_stop ]


################### IMPORT MODULES #######################

import math
import statistics
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Int32, String
from std_srvs.srv import SetBool, Trigger
import tf2_ros


##################### TASK CONSTANTS #######################

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
LIFT_HEIGHT_Z = 0.1500                        # Safe transit height after picking (ore hangs 0.076 m below tool;
                                              # 7 cm clearance over neighbours at z=-0.065)

# Container rim height from the Kepler Colony SDF:
# model.sdf cargo_rover_link z=0.5688 + mesh rim y=0.2814 - mount_z=0.8825
CONTAINER_RIM_Z = -0.0323                     # Measured container rim height in base_link (m)
DROP_DESCENT_HEIGHT_Z = CONTAINER_RIM_Z + 0.0762 + 0.03   # 0.0739 m: Descent height into container for clean release
PRE_DROP_HEIGHT_Z = DROP_DESCENT_HEIGHT_Z + 0.06          # 0.1339 m: Approach height above cargo container
COMFORT_REACH_M = 0.82              # Reach below which no tilt is needed (m)
TOOL_LEN_M = 0.25                   # Effective tool length for tilt calculation (m)
MAX_DROP_TILT_DEG = 35              # Maximum allowed tool tilt toward container (deg)
MAX_DROP_REACH_M = 0.85              # Max XY reach for straight-line container approach (m)
FAR_DROP_PRE_Z = DROP_DESCENT_HEIGHT_Z + 0.02  # ~0.094 m: lower approach for far containers
FAR_DROP_SPEED_MPS = 0.08            # Slower cruise for far container approach (m/s)
FAR_DROP_SLOW_REACH_M = 0.80         # Far approach is slowed only beyond this reach (m)

# The ore package. It stands on the eBot's top deck beside the arm (its top face ~0.25 m above
# base_link), weighs 0.30 kg (force ~2.94 N while held) and is 177.8 x 139.7 x 76.2 mm.
PACKAGE_FRAME = 'ore_package'                 # transform published by the perception node
PACKAGE_KIND = 'ore_package'                  # kind used for the package in the pick queue
PACKAGE_Z_MIN_M = 0.15                        # lowest package top height accepted (m, base_link)
PACKAGE_Z_MAX_M = 0.40                        # highest package top height accepted (m, base_link)
PACKAGE_SETTLE_S = 1.0                        # its transform must be steady this long (s)
PACKAGE_MAX_SPREAD_M = 0.01                   # max per-axis spread over that time (m)
PACKAGE_CLEARANCE_M = 0.15                    # approach and lift height above the package top (m)
PACKAGE_HELD_FORCE_N = 2.0                    # mean force confirming the package is held (N)
PACKAGE_HEIGHT_M = 0.0762                     # the package hangs this far below the tool (m)
# At 1.9 m the camera reads the package top ~1 cm high (seen: 0.264 against ~0.255), so the
# fine descent aims this far below the estimate and stops on contact force, as for the ores.
PACKAGE_OVERSHOOT_M = 0.03
# When the magnet catches the package the force reading jumps in one step from ~0 N to
# 5-16 N (seen: 16.4, 4.8 and 5.2 N): the snap of the catch plus its 2.94 N weight. That
# reading means it is caught, not pressed too hard, so the package keeps descending only
# until the first reading above PACKAGE_CONTACT_FORCE_N, and backs off only above a much
# higher limit. Its fine descent is also slower, to soften the snap.
PACKAGE_CONTACT_FORCE_N = 2.5
PACKAGE_FORCE_LIMIT_N = 40.0
PACKAGE_DESCENT_SPEED_MPS = 0.020

# Ore crusher hopper, from the crusher's collision mesh (map frame, opening x -1.78..-1.33,
# y 1.65..2.10, rim z 1.08), moved into base_link (arm base at map (-0.597, 1.865, 0.8825)).
# The drop point sits 0.16 m inside the near edge, so the whole package lands in the opening.
HOPPER_DROP_XY = (-0.89, 0.01)                # drop point in base_link (m)
HOPPER_RIM_Z = 0.21                           # rim height in base_link (m)
HOPPER_DROP_Z = HOPPER_RIM_Z + PACKAGE_HEIGHT_M + 0.05   # tool height at release: package 5 cm over the rim
HOPPER_PRE_DROP_Z = HOPPER_DROP_Z + 0.06      # approach height above the hopper

# Passed when the tool crosses between the conveyor side and the eBot side, so it never sweeps
# over the base axis (the shoulder singularity). In base_link (m).
SIDE_WAYPOINT = (0.05, -0.45, 0.40)
EBOT_SIDE_X = 0.25                            # the tool is on the eBot side beyond this x (m)

# Motion Speeds & Acceleration Limits
CRUISE_SPEED_MPS = 0.22                       # Main transit linear velocity (m/s); Task 3 allows 0.25
MIN_SERVO_SPEED_MPS = 0.008                   # Minimum speed floor during Cartesian moves
PICK_DESCENT_SPEED_MPS = 0.035                # Slow descent to ore to avoid impact (m/s)
DROP_DESCENT_SPEED_MPS = 0.050                # Gentle descent into container (m/s)
POST_CONTACT_LIFT_SPEED_MPS = 0.050           # Lift speed immediately after contact (m/s)
CAP_LINEAR_MPS = 0.24                         # Safety velocity cap (m/s): the arm drops any command over 0.25
CAP_ANGULAR_RPS = 0.35                        # Safety angular velocity cap (rad/s)
LINEAR_ACCEL_LIMIT = 0.60                     # Max linear acceleration (m/s^2)
ANGULAR_ACCEL_LIMIT = 0.80                    # Max angular acceleration (rad/s^2)

# Control Gains & Tolerances
KP_POSITION = 1.30                            # Proportional gain for Cartesian position error (contact moves)
KP_TRANSIT = 3.00                             # Faster slow-down gain for free-space stops (no contact)
KP_ORIENTATION = 1.00                         # Proportional gain for vertical tool orientation hold
KP_JOINT = 1.50                               # Proportional gain for initial joint unfolding
UNFOLD_MAX_RATE_RPS = 0.55                    # Joint speed cap while unfolding (rad/s); Task 3 allows 0.60
UNFOLD_ACCEL_RPS2 = 0.60                      # Joint acceleration limit while unfolding (rad/s^2); allows 0.8
POSITION_TOLERANCE_M = 0.012                  # General position arrival tolerance (m)
PRE_PICK_TOL_M = 0.004                        # Pre-pick arrival tolerance (m)
DESCENT_TOL_M = 0.003                         # Fine descent arrival tolerance (m)
DESCENT_STAGE1_TOL_M = 0.008                  # Stage-1 descent target tolerance (m): coarser, enters fine stage sooner
PRE_DROP_TOL_M = 0.02                         # Pre-drop approach arrival tolerance (m)
PASS_TOL_M = 0.05                             # Pass-through waypoint tolerance (m) -- no stop, no zero-twist
STANDSTILL_HOLD_S = 0.25                      # Settle pause at waypoints / post contact (s)
FINE_SETTLE_S = 0.4          # hold still at fine-approach height so tool spring settles
CONTACT_CONFIRM_S = 0.10     # force must stay >= CONTACT_FORCE_TRIGGER_N this long
FINE_APPROACH_M = 0.020                       # Height above ore to switch to fine descent (m)
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

# Ore discovery: a transform counts once it has been steady for DISCOVER_SETTLE_S
ORE_Z_MIN_M = -0.085                          # lowest ore top-face height accepted (m, base_link)
ORE_Z_MAX_M = -0.045                          # highest ore top-face height accepted (m, base_link)
DISCOVER_SETTLE_S = 1.0                       # Time window required for stable frames (s)
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
STARTUP_DELAY_S = 1.0                         # Wait time after controllers become ready (s)
CONTROL_PERIOD_S = 0.05                       # 20 Hz main control loop

# Joint configuration the arm unfolds to first, away from joint limits and the wrist singularity
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


##################### MATH UTILITIES #######################

def quat_normalize(quat):
    '''
    Description:    Scales a quaternion to unit length.

    Args:
        quat            (tuple):    (x, y, z, w) quaternion

    Returns:
        unit_q          (tuple):    the unit quaternion, or identity if it has zero length
    '''
    x, y, z, real = quat
    length = math.sqrt(x * x + y * y + z * z + real * real)
    if length < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / length, y / length, z / length, real / length)


def quat_conjugate(quat):
    '''
    Description:    Conjugate (inverse, for a unit quaternion) of a quaternion.

    Args:
        quat            (tuple):    (x, y, z, w) quaternion

    Returns:
        conj_q          (tuple):    (-x, -y, -z, w)
    '''
    x, y, z, real = quat
    return (-x, -y, -z, real)


def quat_multiply(first, second):
    '''
    Description:    Hamilton product first * second (apply second, then first).

    Args:
        first           (tuple):    (x, y, z, w) left quaternion
        second          (tuple):    (x, y, z, w) right quaternion

    Returns:
        product         (tuple):    (x, y, z, w) product
    '''
    first_x, first_y, first_z, first_w = first
    second_x, second_y, second_z, second_w = second
    return (
        first_w * second_x + first_x * second_w + first_y * second_z - first_z * second_y,
        first_w * second_y - first_x * second_z + first_y * second_w + first_z * second_x,
        first_w * second_z + first_x * second_y - first_y * second_x + first_z * second_w,
        first_w * second_w - first_x * second_x - first_y * second_y - first_z * second_z,
    )


def quaternion_error(target_q, current_q):
    '''
    Description:    Rotation that takes the current orientation to the target, as axis and angle.

    Args:
        target_q        (tuple):    (x, y, z, w) orientation to reach
        current_q       (tuple):    (x, y, z, w) orientation now

    Returns:
        error           (tuple):    (axis_x, axis_y, axis_z, angle_rad); all zero if already aligned
    '''
    target_q = quat_normalize(target_q)
    current_q = quat_normalize(current_q)
    error_q = quat_multiply(target_q, quat_conjugate(current_q))
    if error_q[3] < 0.0:
        error_q = tuple(-component for component in error_q)

    x, y, z, real = error_q
    real = max(-1.0, min(1.0, real))
    angle = 2.0 * math.acos(real)
    sin_half = math.sqrt(max(0.0, 1.0 - real * real))
    if sin_half < 1e-8 or angle < 1e-8:
        return 0.0, 0.0, 0.0, 0.0
    return (x / sin_half, y / sin_half, z / sin_half, angle)


def vector_norm(values):
    '''
    Description:    Euclidean length of a vector.

    Args:
        values          (tuple):    vector components

    Returns:
        norm            (float):    length of the vector
    '''
    return math.sqrt(sum(value * value for value in values))


def cap_vector(values, maximum):
    '''
    Description:    Scales a vector down, keeping its direction, so its length is at most 'maximum'.

    Args:
        values          (tuple):    vector components
        maximum         (float):    largest allowed length

    Returns:
        capped          (tuple):    the vector, shortened if it was too long
    '''
    magnitude = vector_norm(values)
    if magnitude <= maximum or magnitude < 1e-12:
        return values
    scale = maximum / magnitude
    return tuple(value * scale for value in values)


def rate_limit_vector(desired, previous, max_delta):
    '''
    Description:    Moves from the previous command towards the desired one by at most
                    'max_delta', which limits the acceleration of the tool.

    Args:
        desired         (tuple):    command wanted this cycle
        previous        (tuple):    command sent last cycle
        max_delta       (float):    largest allowed change in one cycle

    Returns:
        limited         (tuple):    command to send this cycle
    '''
    diff = tuple(desired_component - previous_component
                 for desired_component, previous_component in zip(desired, previous))
    norm = vector_norm(diff)
    if norm <= max_delta:
        return desired
    scale = max_delta / norm
    return tuple(previous_component + delta_component * scale
                 for previous_component, delta_component in zip(previous, diff))


def quat_rotate(quat, vector):
    '''
    Description:    Rotates a 3D vector by a unit quaternion.

    Args:
        quat            (tuple):    (x, y, z, w) unit quaternion
        vector          (tuple):    (x, y, z) vector to rotate

    Returns:
        rotated         (tuple):    (x, y, z) rotated vector
    '''
    vec_x, vec_y, vec_z = vector
    x, y, z, real = quat
    cross_x = 2.0 * (y * vec_z - z * vec_y)
    cross_y = 2.0 * (z * vec_x - x * vec_z)
    cross_z = 2.0 * (x * vec_y - y * vec_x)
    return (
        vec_x + real * cross_x + (y * cross_z - z * cross_y),
        vec_y + real * cross_y + (z * cross_x - x * cross_z),
        vec_z + real * cross_z + (x * cross_y - y * cross_x),
    )


def shortest_arc_quat(v_from, v_to):
    '''
    Description:    Smallest rotation that turns one unit vector onto another.

    Args:
        v_from          (tuple):    (x, y, z) starting unit vector
        v_to            (tuple):    (x, y, z) target unit vector

    Returns:
        rotation        (tuple):    (x, y, z, w) unit quaternion
    '''
    dot = max(-1.0, min(1.0, sum(source_component * target_component
                                 for source_component, target_component in zip(v_from, v_to))))
    if dot > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -0.999999:
        axis = (1.0, 0.0, 0.0) if abs(v_from[0]) < 0.9 else (0.0, 1.0, 0.0)
        cross = (
            axis[1] * v_from[2] - axis[2] * v_from[1],
            axis[2] * v_from[0] - axis[0] * v_from[2],
            axis[0] * v_from[1] - axis[1] * v_from[0],
        )
        cross_len = math.sqrt(sum(component * component for component in cross))
        return (cross[0] / cross_len, cross[1] / cross_len, cross[2] / cross_len, 0.0)
    cross = (
        v_from[1] * v_to[2] - v_from[2] * v_to[1],
        v_from[2] * v_to[0] - v_from[0] * v_to[2],
        v_from[0] * v_to[1] - v_from[1] * v_to[0],
    )
    real = 1.0 + dot
    return quat_normalize((cross[0], cross[1], cross[2], real))


##################### CLASS DEFINITION #######################

class arm_manipulation(Node):
    '''
    ___CLASS___

    Description:    Class which sorts the ores and takes the ore package off the eBot: a
                    timer-driven, non-blocking state machine that reads each item's transform,
                    picks it with the magnet (confirming the grip from the force at the tool),
                    drops each ore in the container for its kind and the package in the ore
                    crusher, tells the eBot when the package is off its deck, and recovers
                    from protective stops.
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
    STATE_WAIT_FOR_PACKAGE = 24

    def __init__(self):
        '''
        Description:    Initialization of class arm_manipulation
        '''
        super().__init__(
            'task3A_manipulation_node',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        ############ Topic PUBLISHERS ############

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.picked_pub = self.create_publisher(String, '/ore_picked', 10)

        # Handover with the eBot: both topics are latched, so a late subscriber still sees them
        latched = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.lifted_pub = self.create_publisher(Bool, '/package_lifted', latched)     # the eBot may leave

        ############ Topic SUBSCRIPTIONS ############

        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)
        self.force_sub = self.create_subscription(Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 20)
        self.at_arm_sub = self.create_subscription(Bool, '/ebot_at_arm', self.ebotatarmcb, latched)

        ############ Service CLIENTS ############

        self.switch_cli = self.create_client(SwitchController, f'{SERVO_NS}/switch_controller')
        self.magnet_cli = self.create_client(SetBool, '/magnet')
        self.unlock_cli = self.create_client(Trigger, f'{SERVO_NS}/unlock_protective_stop')

        ############ Constructor VARIABLES/OBJECTS ############

        # TF buffer and listener, for the ore transforms published by the perception node
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State machine and latest sensor readings
        self.state = self.STATE_STARTUP_WAIT
        self.previous_state_before_latch = self.STATE_MOVE_HOME
        self.tcp_pose = None
        self.joint_angles = {}
        self.arm_status = None
        self.force_magnitude = 0.0

        self.target_orientation = None
        self.previous_linear = (0.0, 0.0, 0.0)
        self.previous_angular = (0.0, 0.0, 0.0)
        self.previous_joint_rates = [0.0] * 6         # last unfold command, for the acceleration ramp
        self.magnet_on = False                        # magnet ON confirmed for the current pick

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
        self.discovery_history = {}  # frame id -> [(stamp, x, y, z)]

        # Ore package handover
        self.ebot_at_arm = False                # from ebotatarmcb()
        self.package_history = []               # [(t, x, y, z)] samples of the package transform
        self.package_pose = None                # steady package top-face position in base_link
        self.package_failed = False             # True once the package has run out of pick attempts
        self.lift_reported = False              # /package_lifted already published
        self.side_passed = False                # SIDE_WAYPOINT reached on the current crossing
        self.exit_stage = None                  # leave_ebot_side() progress: None, 'climb' or 'side'
        self.exit_xy = None                     # where the climb out of the eBot side started

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

        # One handler method per state, run by step_state_machine()
        self.state_handlers = {
            self.STATE_STARTUP_WAIT: self.state_startup_wait,
            self.STATE_SWITCH_TO_JOINT: self.state_switch_to_joint,
            self.STATE_UNFOLD_JOINTS: self.state_unfold_joints,
            self.STATE_SWITCH_TO_TWIST: self.state_switch_to_twist,
            self.STATE_MOVE_HOME: self.state_move_home,
            self.STATE_DISCOVER_ORES: self.state_discover_ores,
            self.STATE_APPROACH_PRE_PICK: self.state_approach_pre_pick,
            self.STATE_ENABLE_MAGNET: self.state_enable_magnet,
            self.STATE_DESCENT_PICK: self.state_descent_pick,
            self.STATE_DESCENT_BACKOFF: self.state_descent_backoff,
            self.STATE_CONTACT_HOLD: self.state_contact_hold,
            self.STATE_CONTACT_LIFT: self.state_contact_lift,
            self.STATE_CONFIRM_GRIP: self.state_confirm_grip,
            self.STATE_LIFT_AFTER_PICK: self.state_lift_after_pick,
            self.STATE_TRANSIT_HOME_WITH_ORE: self.state_transit_home_with_ore,
            self.STATE_APPROACH_PRE_DROP: self.state_approach_pre_drop,
            self.STATE_DESCENT_DROP: self.state_descent_drop,
            self.STATE_DISABLE_MAGNET: self.state_disable_magnet,
            self.STATE_CONFIRM_RELEASE: self.state_confirm_release,
            self.STATE_RETREAT_TO_PRE_DROP: self.state_retreat_to_pre_drop,
            self.STATE_RETURN_HOME: self.state_return_home,
            self.STATE_LATCH_RECOVERY_UNLOCK: self.state_latch_recovery_unlock,
            self.STATE_LATCH_RECOVERY_CLEAR: self.state_latch_recovery_clear,
            self.STATE_ALL_COMPLETED: self.state_all_completed,
            self.STATE_WAIT_FOR_PACKAGE: self.state_wait_for_package,
        }

        # Main Control Timer (20 Hz)
        self.timer = self.create_timer(CONTROL_PERIOD_S, self.step_state_machine)
        self.get_logger().info('Task 3A Manipulation Node initialized.')
        self.get_logger().info(
            f'Cargo Container Heights: rim_z={CONTAINER_RIM_Z:.4f}m, '
            f'DROP_DESCENT_HEIGHT_Z={DROP_DESCENT_HEIGHT_Z:.4f}m, '
            f'PRE_DROP_HEIGHT_Z={PRE_DROP_HEIGHT_Z:.4f}m'
        )

    def tcpposecb(self, data):
        '''
        Description:    Callback function for the tool pose topic. Stores the latest tcp pose.

        Args:
            data        (PoseStamped):  tcp pose in base_link

        Returns:
        '''
        self.tcp_pose = data.pose
        if not self.logged_tcp_frame:
            self.get_logger().info(f'/tcp_pose_raw header.frame_id: {data.header.frame_id}')
            self.logged_tcp_frame = True

    def jointstatecb(self, data):
        '''
        Description:    Callback function for the joint states topic. Stores each joint's angle.

        Args:
            data        (JointState):   names and positions of the arm's joints

        Returns:
        '''
        for name, pos in zip(data.name, data.position):
            self.joint_angles[name] = pos

    def armstatuscb(self, data):
        '''
        Description:    Callback function for the arm status topic. Stores the safety code and
                        logs every change.

        Args:
            data        (Int32):    arm status code (20 to 23 are protective stops)

        Returns:
        '''
        prev = self.arm_status
        self.arm_status = data.data
        if prev != self.arm_status:
            self.get_logger().info(f'Arm status changed: {prev} -> {self.arm_status}')

    def forcecb(self, data):
        '''
        Description:    Callback function for the tool force topic. Stores the force and keeps
                        the last second of readings for averaging.

        Args:
            data        (Float32):  force at the tool, in newtons

        Returns:
        '''
        self.force_magnitude = float(data.data)
        now = self.get_clock().now().nanoseconds / 1e9
        self.recent_forces.append((now, self.force_magnitude))
        self.recent_forces = [sample for sample in self.recent_forces if now - sample[0] <= 1.0]

    def ebotatarmcb(self, data):
        '''
        Description:    Callback function for /ebot_at_arm. The planner reports that the eBot
                        stands beside the arm with the ore package on its deck.

        Args:
            data        (Bool):     True once the eBot has stopped beside the arm

        Returns:
        '''
        if data.data and not self.ebot_at_arm:
            self.ebot_at_arm = True
            self.get_logger().info('eBot beside the arm: the package will be taken next')

    def update_package(self, now):
        '''
        Description:    Samples the ore package transform and, once it has been steady for
                        PACKAGE_SETTLE_S, stores its median as the pick target.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if not self.ebot_at_arm or PACKAGE_KIND in self.done_ores:
            return
        try:
            package_tf = self.tf_buffer.lookup_transform(BASE_FRAME, PACKAGE_FRAME, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        stamp = package_tf.header.stamp.sec + package_tf.header.stamp.nanosec * 1e-9
        if not 0.0 <= now - stamp < DISCOVER_MAX_TF_AGE_S:
            return
        position = package_tf.transform.translation
        if not PACKAGE_Z_MIN_M <= position.z <= PACKAGE_Z_MAX_M:
            return
        self.package_history.append((now, position.x, position.y, position.z))
        self.package_history = [sample for sample in self.package_history if now - sample[0] <= PACKAGE_SETTLE_S]
        if (self.package_history[-1][0] - self.package_history[0][0]) < PACKAGE_SETTLE_S - 0.2:
            return
        axes = list(zip(*self.package_history))[1:]
        if all(max(values) - min(values) < PACKAGE_MAX_SPREAD_M for values in axes):
            self.package_pose = tuple(float(statistics.median(values)) for values in axes)

    def package_ready(self):
        '''
        Description:    True when the package should be picked next: the eBot is beside the
                        arm, the package position is steady, and it is not done or abandoned.

        Args:

        Returns:
            ready           (bool):     True if the package can be picked now
        '''
        return (self.ebot_at_arm and self.package_pose is not None and not self.package_failed
                and PACKAGE_KIND not in self.done_ores)

    def is_package(self):
        '''
        Description:    True when the item being handled is the ore package.

        Args:

        Returns:
            package         (bool):     True for the package, False for an ore
        '''
        return self.current_ore is not None and self.current_ore[1] == PACKAGE_KIND

    def next_target(self, now):
        '''
        Description:    Chooses the next item and enters the matching state. The package comes
                        first as soon as it is ready, so the eBot can drive home while the arm
                        sorts the remaining ores. With no ore left, the arm waits for the package.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        self.ore_queue = [ore for ore in self.ore_queue if ore[0] not in self.done_ores and ore[1] != PACKAGE_KIND]
        self.pick_attempt_count = 0
        self.side_passed = False
        if self.package_ready():
            self.current_ore = (PACKAGE_FRAME, PACKAGE_KIND, self.package_pose)
            self.get_logger().info(f'Next: the ore package at {tuple(round(value, 3) for value in self.package_pose)}')
            self.change_state(self.STATE_APPROACH_PRE_PICK, now)
        elif self.ore_queue:
            self.current_ore = self.ore_queue.pop(0)
            self.change_state(self.STATE_APPROACH_PRE_PICK, now)
        elif PACKAGE_KIND not in self.done_ores and not self.package_failed:
            self.current_ore = None
            self.change_state(self.STATE_WAIT_FOR_PACKAGE, now)
        else:
            self.get_logger().info('All completed!')
            self.change_state(self.STATE_ALL_COMPLETED, now)

    def give_up_current(self, now):
        '''
        Description:    The current item ran out of pick attempts: an ore is retried once at
                        the end of the queue, the package is abandoned. Then moves on.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if self.is_package():
            self.get_logger().error('Ore package could not be picked; continuing with the ores.')
            self.package_failed = True
        elif self.current_ore[0] not in self.second_pass_ores:
            self.get_logger().warn(f'Max pick attempts exceeded for {self.current_ore[0]}. Retrying it last.')
            self.second_pass_ores.add(self.current_ore[0])
            self.ore_queue.append(self.current_ore)
        self.next_target(now)

    def pick_heights(self):
        '''
        Description:    Approach and lift heights for the current item: fixed for the ores on
                        the conveyor, PACKAGE_CLEARANCE_M above the package top for the package.

        Args:

        Returns:
            heights         (tuple):    (pre-pick z, lift z) in base_link, in metres
        '''
        if self.is_package():
            top_z = self.current_ore[2][2]
            return (top_z + PACKAGE_CLEARANCE_M, top_z + PACKAGE_CLEARANCE_M)
        return (PRE_PICK_HEIGHT_Z, LIFT_HEIGHT_Z)

    def drop_plan(self):
        '''
        Description:    Where and how the current item is dropped: the crusher hopper for the
                        package, the container for its kind for an ore.

        Args:

        Returns:
            plan            (tuple):    (x, y, approach z, release z, approach speed, far) where
                                        'far' means retreat straight to SAFE_HOME afterwards
        '''
        if self.is_package():
            return (HOPPER_DROP_XY[0], HOPPER_DROP_XY[1], HOPPER_PRE_DROP_Z, HOPPER_DROP_Z,
                    CRUISE_SPEED_MPS, False)
        drop_x, drop_y = self.drop_xy(self.current_ore[1])
        is_far = math.hypot(drop_x, drop_y) > MAX_DROP_REACH_M
        pre_z = FAR_DROP_PRE_Z if is_far else PRE_DROP_HEIGHT_Z
        speed = FAR_DROP_SPEED_MPS if is_far else CRUISE_SPEED_MPS
        return (drop_x, drop_y, pre_z, DROP_DESCENT_HEIGHT_Z, speed, is_far)

    def via_side(self, target_xyz, now, through=True):
        '''
        Description:    Moves towards target_xyz, first through SIDE_WAYPOINT when the package
                        is handled, so the tool never passes over the arm's base.

        Args:
            target_xyz      (tuple):    (x, y, z) final point in base_link, in metres
            now             (float):    current time in seconds
            through         (bool):     pass through target_xyz without stopping

        Returns:
            arrived         (bool):     True once target_xyz is reached
        '''
        if self.is_package() and not self.side_passed:
            if self.navigate_cartesian(SIDE_WAYPOINT, CRUISE_SPEED_MPS, now, through=True):
                self.side_passed = True
            return False
        return self.navigate_cartesian(target_xyz, CRUISE_SPEED_MPS, now, through=through)

    def leave_ebot_side(self, now):
        '''
        Description:    Brings the tool back from the eBot side before any other move: straight
                        up to the side waypoint's height, then through SIDE_WAYPOINT, so it
                        never sweeps low across the eBot or over the arm's base.

        Args:
            now             (float):    current time in seconds

        Returns:
            clear           (bool):     True once the tool is off the eBot side
        '''
        if self.exit_stage is None:
            if self.tcp_pose.position.x <= EBOT_SIDE_X:
                return True
            self.exit_stage = 'climb'
            self.exit_xy = (self.tcp_pose.position.x, self.tcp_pose.position.y)
        if self.exit_stage == 'climb':
            if self.navigate_cartesian((self.exit_xy[0], self.exit_xy[1], SIDE_WAYPOINT[2]),
                                       CRUISE_SPEED_MPS, now, through=True):
                self.exit_stage = 'side'
            return False
        if self.navigate_cartesian(SIDE_WAYPOINT, CRUISE_SPEED_MPS, now, through=True):
            self.exit_stage = None
            return True
        return False

    def change_state(self, new_state, now):
        '''
        Description:    Moves the state machine to a new state, logs the transition, and resets
                        the latch counter when leaving latch recovery.

        Args:
            new_state       (int):      state to enter (one of the STATE_ constants)
            now             (float):    current time in seconds

        Returns:
        '''
        old_state = self.state
        latch_states = (self.STATE_LATCH_RECOVERY_UNLOCK, self.STATE_LATCH_RECOVERY_CLEAR)
        if new_state != old_state and old_state not in latch_states and new_state not in latch_states:
            self.latch_count = 0

        if new_state == self.STATE_APPROACH_PRE_PICK:
            self.magnet_on = False
            self.magnet_future = None
            self.magnet_retry_count = 0
            self.last_magnet_call_time = 0.0

        self.get_logger().info(
            f'[STATE TRANSITION] {self.state_name(self.state)} -> {self.state_name(new_state)} '
            f'| Ore: {self.current_ore[0] if self.current_ore else "None"} '
            f'| Force: {self.force_magnitude:.2f}N | Status: {self.arm_status}'
        )
        self.state = new_state
        self.state_enter_time = now

    def state_name(self, state_id):
        '''
        Description:    Readable name of a state, for the logs.

        Args:
            state_id        (int):      state number

        Returns:
            name            (str):      the state's name without the STATE_ prefix
        '''
        for attr_name, attr_value in self.__class__.__dict__.items():
            if attr_name.startswith('STATE_') and attr_value == state_id:
                return attr_name[6:]
        return str(state_id)

    def switch_controller(self, activate_name, deactivate_name=None):
        '''
        Description:    Asks the arm to switch its active input controller, without blocking.

        Args:
            activate_name       (str):  controller to activate
            deactivate_name     (str):  controller to deactivate, or None

        Returns:
            sent            (bool):     True if the request was sent
        '''
        if not self.switch_cli.service_is_ready():
            return False
        request = SwitchController.Request()
        request.activate_controllers = [activate_name]
        if deactivate_name:
            request.deactivate_controllers = [deactivate_name]
        request.strictness = SwitchController.Request.BEST_EFFORT
        self.switch_future = self.switch_cli.call_async(request)
        return True

    def call_magnet(self, enable: bool):
        '''
        Description:    Switches the magnet on or off, without blocking.

        Args:
            enable          (bool):     True to switch the magnet on, False to release

        Returns:
            sent            (bool):     True if the request was sent
        '''
        if not self.magnet_cli.service_is_ready():
            return False
        request = SetBool.Request()
        request.data = enable
        self.magnet_future = self.magnet_cli.call_async(request)
        return True

    def poll_magnet_on(self, now):
        '''
        Description:    Switches the magnet on during the ore descent and tracks the reply,
                        retrying a failed call. Lets the descent continue meanwhile.

        Args:
            now             (float):    current time in seconds

        Returns:
            confirmed       (bool):     True once the magnet has reported it is on
        '''
        if self.magnet_on:
            return True
        if self.magnet_future is None:
            if (now - self.last_magnet_call_time) >= MAGNET_RETRY_INTERVAL_S:
                self.call_magnet(True)
                self.magnet_retry_count += 1
                self.last_magnet_call_time = now
                if self.magnet_retry_count > MAGNET_MAX_RETRIES:
                    self.get_logger().error('Magnet ON failed after max retries; descending anyway.')
                    self.magnet_on = True
            return self.magnet_on
        if self.magnet_future.done():
            try:
                self.magnet_on = bool(self.magnet_future.result().success)
            except Exception as error:
                self.get_logger().error(f'Magnet service error: {error}')
            self.magnet_future = None
        return self.magnet_on

    def call_unlock(self):
        '''
        Description:    Asks the arm controller to unlock a protective stop, without blocking.

        Args:

        Returns:
            sent            (bool):     True if the request was sent
        '''
        if not self.unlock_cli.service_is_ready():
            return False
        request = Trigger.Request()
        self.unlock_future = self.unlock_cli.call_async(request)
        return True

    def publish_zero_twist(self):
        '''
        Description:    Commands zero tool velocity and resets the acceleration limiter.

        Args:

        Returns:
        '''
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        self.twist_pub.publish(msg)
        self.previous_linear = (0.0, 0.0, 0.0)
        self.previous_angular = (0.0, 0.0, 0.0)

    def publish_zero_joints(self):
        '''
        Description:    Commands zero velocity on every joint.

        Args:

        Returns:
        '''
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = JOINT_NAMES
        msg.velocities = [0.0] * len(JOINT_NAMES)
        self.joint_pub.publish(msg)

    def tilted_orientation(self, x, y):
        '''
        Description:    Tool orientation for a drop at (x, y): straight down when within comfortable
                        reach, otherwise tilted towards the target by up to MAX_DROP_TILT_DEG.

        Args:
            x               (float):    drop point x in base_link, in metres
            y               (float):    drop point y in base_link, in metres

        Returns:
            orientation     (tuple):    (x, y, z, w) target orientation of the tool
        '''
        reach = math.hypot(x, y)
        if reach < 1e-6 or self.vertical_orientation is None:
            return self.vertical_orientation
        excess = max(0.0, reach - COMFORT_REACH_M)
        tilt = min(max(0.0, math.asin(min(1.0, excess / TOOL_LEN_M))),
                   math.radians(MAX_DROP_TILT_DEG))
        if tilt < 1e-4:
            return self.vertical_orientation
        unit_x, unit_y = x / reach, y / reach
        tool_dir = (unit_x * math.sin(tilt), unit_y * math.sin(tilt), -math.cos(tilt))
        q_delta = shortest_arc_quat((0.0, 0.0, -1.0), tool_dir)
        return quat_normalize(quat_multiply(q_delta, self.vertical_orientation))

    def drop_xy(self, kind):
        '''
        Description:    Drop point (x, y) above the container for a kind of ore, logged once per kind.
                        The package uses HOPPER_DROP_XY instead (see drop_plan()).

        Args:
            kind            (str):      kind of ore, e.g. 'azurite_ore'

        Returns:
            point           (tuple):    (x, y) of the container centre in base_link, in metres
        '''
        if kind in self.drop_xy_logged:
            return self.drop_xy_logged[kind]
        bay_x, bay_y, _ = CONTAINER_DROP_POSES[kind]
        dist = math.hypot(bay_x, bay_y)
        self.get_logger().info(f'[DROP-XY] {kind}: centre ({bay_x:.4f},{bay_y:.4f}) dist={dist:.4f}m')
        result = (bay_x, bay_y)
        self.drop_xy_logged[kind] = result
        return result

    def navigate_cartesian(self, target_xyz, cruise_speed, now, tol=POSITION_TOLERANCE_M, through=False,
                           gain=KP_POSITION):
        '''
        Description:    One control cycle servoing the tcp towards a point in base_link, holding the
                        target orientation, with speed and acceleration limits.

        Args:
            target_xyz      (tuple):    (x, y, z) point to reach, in metres
            cruise_speed    (float):    top speed for this move, in m/s
            now             (float):    current time in seconds
            tol             (float):    arrival tolerance in metres
            through         (bool):     True for a pass-through waypoint: no slow-down, no stop
            gain            (float):    slow-down gain near the target, KP_TRANSIT in free space

        Returns:
            arrived         (bool):     True once within the tolerance
        '''
        tool_x, tool_y, tool_z = self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z
        goal_x, goal_y, goal_z = target_xyz
        error_vec = (goal_x - tool_x, goal_y - tool_y, goal_z - tool_z)
        dist = vector_norm(error_vec)

        arrival_tol = PASS_TOL_M if through else tol
        if dist <= arrival_tol:
            if not through:
                self.publish_zero_twist()
                self.previous_linear = (0.0, 0.0, 0.0)
                self.previous_angular = (0.0, 0.0, 0.0)
            return True

        unit_dir = tuple(component / dist for component in error_vec)
        if through:
            speed = cruise_speed  # no proportional slow-down for pass-through
        else:
            speed = max(MIN_SERVO_SPEED_MPS, min(cruise_speed, gain * dist))
        desired_linear = cap_vector(tuple(component * speed for component in unit_dir), CAP_LINEAR_MPS)

        max_linear_step = LINEAR_ACCEL_LIMIT * CONTROL_PERIOD_S
        linear_cmd = rate_limit_vector(desired_linear, self.previous_linear, max_linear_step)

        # Orientation stabilization to downward vertical
        current_q = quat_normalize((
            self.tcp_pose.orientation.x,
            self.tcp_pose.orientation.y,
            self.tcp_pose.orientation.z,
            self.tcp_pose.orientation.w,
        ))
        axis_x, axis_y, axis_z, angle = quaternion_error(self.target_orientation, current_q)
        desired_angular = cap_vector(
            (axis_x * angle * KP_ORIENTATION, axis_y * angle * KP_ORIENTATION, axis_z * angle * KP_ORIENTATION),
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

    def tcp_text(self):
        '''
        Description:    The tool position as text, for the logs.

        Args:

        Returns:
            text            (str):      '(x, y, z)' in base_link, or 'None' before the first pose
        '''
        if not self.tcp_pose:
            return 'None'
        position = self.tcp_pose.position
        return f'({position.x:.4f}, {position.y:.4f}, {position.z:.4f})'

    def joints_text(self):
        '''
        Description:    Every joint angle as text, for the logs.

        Args:

        Returns:
            text            (str):      'name: angle, ...' in radians
        '''
        return ', '.join(f'{joint}: {self.joint_angles.get(joint, 0.0):.3f}' for joint in JOINT_NAMES)

    def step_state_machine(self):
        '''
        Description:    Timer function running one step of the state machine: refreshes the
                        package estimate, checks for a protective stop, then runs the handler of
                        the current state (see the state_* methods below).

        Args:

        Returns:
        '''
        now = self.get_clock().now().nanoseconds / 1e9

        if self.node_init_time is None:
            self.node_init_time = now

        # Keep the package estimate fresh while the eBot waits beside the arm
        self.update_package(now)

        # The ores are watched from start-up, so they are known before the arm gets home
        if not self.discovered and self.state != self.STATE_DISCOVER_ORES:
            self.sample_ore_transforms(now)

        if self.check_protective_stop(now):
            return

        handler = self.state_handlers.get(self.state)
        if handler is not None:
            handler(now)

    def check_protective_stop(self, now):
        '''
        Description:    Protective stop monitor (arm status codes 20 to 23). On a new latch it
                        stops the tool, logs where it happened, and starts the recovery states.

        Args:
            now             (float):    current time in seconds

        Returns:
            latched         (bool):     True if a new protective stop was just handled
        '''
        if self.arm_status is None or not 20 <= self.arm_status <= 23:
            return False

        if self.state not in (self.STATE_LATCH_RECOVERY_UNLOCK, self.STATE_LATCH_RECOVERY_CLEAR):
            self.latch_count += 1
            tcp_str, joints_str = self.tcp_text(), self.joints_text()
            ore_name = self.current_ore[0] if self.current_ore else 'None'
            target_xyz = ('({:.4f}, {:.4f}, {:.4f})'.format(*self.current_ore[2])
                          if self.current_ore else 'None')
            self.get_logger().error(
                f'PROTECTIVE STOP DETECTED: code={self.arm_status} (latch_count={self.latch_count}). '
                f'State: {self.state_name(self.state)}, TCP: {tcp_str}, Joints: [{joints_str}], '
                f'Ore: {ore_name}, Target: {target_xyz}. Latching position and initiating recovery...'
            )
            self.get_logger().info(
                f'[DROP-JOINTS] Ore: {self.current_ore[1] if self.current_ore else "None"}, '
                f'TCP: {tcp_str}, Joints: [{joints_str}]'
            )
            self.publish_zero_twist()
            self.previous_state_before_latch = self.state
            self.change_state(self.STATE_LATCH_RECOVERY_UNLOCK, now)
            self.call_unlock()
            return True
        return False

    def state_startup_wait(self, now):
        '''
        Description:    Wait for controller & sensor readiness.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if self.tcp_pose is None or len(self.joint_angles) < 6 or self.arm_status is None:
            return
        if self.arm_status != 0:
            return
        if (now - self.node_init_time) < STARTUP_DELAY_S:
            return

        self.get_logger().info('Arm controller ready. Requesting switch to joint controller for unfold...')
        self.change_state(self.STATE_SWITCH_TO_JOINT, now)
        self.switch_controller(JOINT_CONTROLLER)

    def state_switch_to_joint(self, now):
        '''
        Description:    Switch to delta_joint_controller.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if self.switch_future is not None and self.switch_future.done():
            try:
                response = self.switch_future.result()
                if response.ok:
                    self.get_logger().info('Switched to joint controller. Unfolding joints...')
                    self.change_state(self.STATE_UNFOLD_JOINTS, now)
                else:
                    self.get_logger().warn('Switch to joint controller rejected; retrying...')
                    self.switch_controller(JOINT_CONTROLLER)
            except Exception as error:
                self.get_logger().error(f'Switch error: {error}')
            self.switch_future = None

    def state_unfold_joints(self, now):
        '''
        Description:    Unfold joints to clear singularity & limits.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        max_err = 0.0
        cmd_velocities = []
        max_step = UNFOLD_ACCEL_RPS2 * CONTROL_PERIOD_S
        for index, name in enumerate(JOINT_NAMES):
            curr = self.joint_angles.get(name, 0.0)
            target_angle = UNFOLD_TARGETS[name]
            err = target_angle - curr
            max_err = max(max_err, abs(err))
            wanted = max(-UNFOLD_MAX_RATE_RPS, min(UNFOLD_MAX_RATE_RPS, KP_JOINT * err))
            previous = self.previous_joint_rates[index]
            vel = previous + max(-max_step, min(max_step, wanted - previous))
            cmd_velocities.append(vel)
        self.previous_joint_rates = cmd_velocities

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = JOINT_NAMES
        msg.velocities = cmd_velocities
        self.joint_pub.publish(msg)

        if max_err <= 0.04:
            self.previous_joint_rates = [0.0] * 6
            self.publish_zero_joints()
            self.get_logger().info('Joint unfold complete. Switching to Cartesian twist controller...')
            self.change_state(self.STATE_SWITCH_TO_TWIST, now)
            self.switch_controller(TWIST_CONTROLLER)

    def state_switch_to_twist(self, now):
        '''
        Description:    Switch back to delta_twist_controller.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if self.switch_future is not None and self.switch_future.done():
            try:
                response = self.switch_future.result()
                if response.ok:
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
                        self.get_logger().info(
                            'Built target_orientation via shortest-arc rotation to tool z = (0,0,-1).')
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
            except Exception as error:
                self.get_logger().error(f'Switch error: {error}')
            self.switch_future = None

    def state_move_home(self, now):
        '''
        Description:    Move to Safe Home Waypoint.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        # No stop at home when the ores are known: either queued already (back from a latch
        # recovery), or all seen steady since start-up
        stable = {} if self.discovered else self.stable_ores(now)
        use_through = self.discovered or len(stable) >= DISCOVER_REQUIRED_COUNT
        if not self.leave_ebot_side(now):
            return
        arrived = self.navigate_cartesian(SAFE_HOME_POSE, CRUISE_SPEED_MPS, now, through=use_through,
                                          gain=KP_TRANSIT)
        if arrived:
            if self.discovered:
                self.next_target(now)
            elif use_through:
                self.queue_ores(stable, now)
            else:
                self.change_state(self.STATE_DISCOVER_ORES, now)

    def sample_ore_transforms(self, now):
        '''
        Description:    Adds the current position of every fresh ore transform to the discovery
                        history. Runs during discovery, and already while the arm moves home.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        node_time_sec = self.get_clock().now().nanoseconds / 1e9
        candidate_kinds = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

        for kind in candidate_kinds:
            for idx in range(1, 5):
                frame_id = f'{kind}_{idx}'
                try:
                    transform = self.tf_buffer.lookup_transform(BASE_FRAME, frame_id, Time())
                    tf_sec = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
                    age = node_time_sec - tf_sec
                    if 0.0 <= age < DISCOVER_MAX_TF_AGE_S:
                        ore_x = transform.transform.translation.x
                        ore_y = transform.transform.translation.y
                        ore_z = transform.transform.translation.z
                        if frame_id not in self.discovery_history:
                            self.discovery_history[frame_id] = []
                        self.discovery_history[frame_id].append((now, ore_x, ore_y, ore_z))
                except Exception:
                    pass

    def stable_ores(self, now):
        '''
        Description:    The ores whose transform has stayed put over the last DISCOVER_SETTLE_S,
                        each with its median position.

        Args:
            now             (float):    current time in seconds

        Returns:
            stable          (dict):     frame name -> (frame name, kind, (x, y, z) in base_link)
        '''
        stable = {}
        for fid, samples in self.discovery_history.items():
            recent = [sample for sample in samples if now - sample[0] <= DISCOVER_SETTLE_S]
            self.discovery_history[fid] = recent
            if len(recent) >= 5 and (recent[-1][0] - recent[0][0]) >= (DISCOVER_SETTLE_S - 0.2):
                if (now - recent[-1][0]) < DISCOVER_MAX_TF_AGE_S:
                    x_values = [sample[1] for sample in recent]
                    y_values = [sample[2] for sample in recent]
                    z_values = [sample[3] for sample in recent]
                    if (max(x_values) - min(x_values) < DISCOVER_MAX_SPREAD_M and
                        max(y_values) - min(y_values) < DISCOVER_MAX_SPREAD_M and
                        max(z_values) - min(z_values) < DISCOVER_MAX_SPREAD_M):
                        med_x = float(statistics.median(x_values))
                        med_y = float(statistics.median(y_values))
                        med_z = float(statistics.median(z_values))
                        kind = fid.rsplit('_', 1)[0]
                        if ORE_Z_MIN_M <= med_z <= ORE_Z_MAX_M:
                            stable[fid] = (fid, kind, (med_x, med_y, med_z))
        return stable

    def queue_ores(self, stable, now):
        '''
        Description:    Queues the discovered ores, nearest to the arm first, and starts on
                        the first item.

        Args:
            stable          (dict):     ores from stable_ores()
            now             (float):    current time in seconds

        Returns:
        '''
        ore_list = list(stable.values())
        # Order by xy distance from base_link origin, nearest first
        ore_list.sort(key=lambda item: math.hypot(item[2][0], item[2][1]))
        self.ore_queue = ore_list
        self.discovered = True
        self.get_logger().info(f'Discovered {len(self.ore_queue)} ores: {[ore[0] for ore in self.ore_queue]}')
        self.next_target(now)

    def state_discover_ores(self, now):
        '''
        Description:    Discover Available Ores via TF Buffer.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if self.discovery_start_time is None:
            self.discovery_start_time = now

        self.sample_ore_transforms(now)
        stable = self.stable_ores(now)

        timeout_expired = (now - self.discovery_start_time) >= DISCOVER_TIMEOUT_S
        if len(stable) >= DISCOVER_REQUIRED_COUNT or (timeout_expired and len(stable) >= 1):
            self.queue_ores(stable, now)
        else:
            if now - self.last_log_time > 1.0:
                self.get_logger().info(
                    f'Sampling ores: {len(stable)} stable, time={now - self.discovery_start_time:.1f}s'
                )
                self.last_log_time = now

    def state_approach_pre_pick(self, now):
        '''
        Description:    Move to Pre-Pick Pose (above the ore, or above the package on the eBot).

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if self.vertical_orientation is not None:
            self.target_orientation = self.vertical_orientation
        ore_x, ore_y, _ = self.current_ore[2]
        target_pre_pick = (ore_x, ore_y, self.pick_heights()[0])
        if not self.is_package() and not self.leave_ebot_side(now):
            return
        if self.is_package() and not self.side_passed:
            self.via_side(target_pre_pick, now)
            return
        # Ores: magnet on from the start of the approach, then pass over the pre-pick point
        # and keep going down; the package stops there and switches the magnet on
        through = not self.is_package()
        if through and self.tcp_pose.position.z >= target_pre_pick[2] - 0.05:
            self.poll_magnet_on(now)                  # on a retry, only once clear of the ore
        arrived = self.navigate_cartesian(target_pre_pick, CRUISE_SPEED_MPS, now, tol=PRE_PICK_TOL_M,
                                          gain=KP_TRANSIT, through=through)
        if arrived:
            self.fine_stage = False
            self.fine_enter_time = None
            self.contact_start = None
            if through:
                self.last_descent_log_time = 0.0
                self.change_state(self.STATE_DESCENT_PICK, now)
            else:
                self.magnet_retry_count = 0
                self.last_magnet_call_time = 0.0
                self.magnet_future = None
                self.magnet_on = False
                self.change_state(self.STATE_ENABLE_MAGNET, now)

    def state_enable_magnet(self, now):
        '''
        Description:    Enable Electromagnet on Pre-Pick Arrival.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if self.magnet_future is None:
            if (now - self.last_magnet_call_time) >= MAGNET_RETRY_INTERVAL_S:
                if self.magnet_cli.service_is_ready():
                    self.call_magnet(True)
                self.magnet_retry_count += 1
                self.last_magnet_call_time = now
                if self.magnet_retry_count > MAGNET_MAX_RETRIES:
                    self.get_logger().error('Magnet ON failed after max retries; proceeding to descent.')
                    self.magnet_on = True
                    self.change_state(self.STATE_DESCENT_PICK, now)
        else:
            if self.magnet_future.done():
                try:
                    response = self.magnet_future.result()
                    if response.success:
                        self.magnet_future = None
                        self.magnet_on = True
                        self.last_descent_log_time = 0.0
                        self.change_state(self.STATE_DESCENT_PICK, now)
                        return
                    else:
                        self.magnet_future = None
                except Exception as error:
                    self.get_logger().error(f'Magnet service error: {error}')
                    self.magnet_future = None

    def state_descent_pick(self, now):
        '''
        Description:    Two-Stage Descent to Ore with Force Monitoring.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        ore_x, ore_y, ore_z = self.current_ore[2]
        tool_z = self.tcp_pose.position.z

        if (now - self.last_descent_log_time) >= DESCENT_LOG_INTERVAL_S:
            self.get_logger().info(f'Descent telemetry: z={tool_z:.4f}m, oz={ore_z:.4f}m, '
                                   f'force={self.force_magnitude:.2f}N')
            self.last_descent_log_time = now

        magnet_ready = self.poll_magnet_on(now)
        armed = (self.fine_stage and self.fine_enter_time is not None
                 and (now - self.fine_enter_time) >= FINE_SETTLE_S and magnet_ready)

        force_limit = PACKAGE_FORCE_LIMIT_N if self.is_package() else FORCE_PROTECTIVE_LIMIT_N
        if armed and self.force_magnitude >= force_limit:
            self.publish_zero_twist()
            self.get_logger().warn(
                f'Force exceeded protective limit ({self.force_magnitude:.2f}N >= {force_limit}N). '
                f'Moving up 2 cm and counting failed attempt.'
            )
            self.pick_attempt_count += 1
            self.contact_z = tool_z
            self.change_state(self.STATE_DESCENT_BACKOFF, now)
            return

        contact_n = PACKAGE_CONTACT_FORCE_N if self.is_package() else CONTACT_FORCE_TRIGGER_N
        if armed and self.force_magnitude >= contact_n:
            if self.contact_start is None:
                self.contact_start = now
            confirm_s = 0.0 if self.is_package() else CONTACT_CONFIRM_S
            if (now - self.contact_start) >= confirm_s:
                self.publish_zero_twist()
                self.contact_z = tool_z
                self.get_logger().info(
                    f'Contact detected on {self.current_ore[0]}: force={self.force_magnitude:.2f}N '
                    f'at z={tool_z:.4f}m (ore top z={ore_z:.4f})')
                self.change_state(self.STATE_CONTACT_HOLD, now)
                return
        else:
            self.contact_start = None

        if not self.fine_stage:
            if self.navigate_cartesian((ore_x, ore_y, ore_z + FINE_APPROACH_M), CRUISE_SPEED_MPS, now,
                                       tol=DESCENT_STAGE1_TOL_M, gain=KP_TRANSIT):
                self.fine_stage = True
                self.fine_enter_time = now
                self.publish_zero_twist()
        elif not armed:
            self.publish_zero_twist()
        else:
            overshoot = PACKAGE_OVERSHOOT_M if self.is_package() else PICK_OVERSHOOT_M
            speed = PACKAGE_DESCENT_SPEED_MPS if self.is_package() else PICK_DESCENT_SPEED_MPS
            arrived = self.navigate_cartesian((ore_x, ore_y, ore_z - overshoot), speed, now,
                                              tol=DESCENT_TOL_M)
            if arrived:
                self.publish_zero_twist()
                self.contact_z = tool_z
                self.change_state(self.STATE_CONTACT_HOLD, now)

    def state_descent_backoff(self, now):
        '''
        Description:    Descent Safety Backoff (move up 2 cm).

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        ore_x, ore_y, _ = self.current_ore[2]
        target_backoff = (ore_x, ore_y, self.contact_z + 0.02)
        arrived = self.navigate_cartesian(target_backoff, PICK_DESCENT_SPEED_MPS, now, tol=POSITION_TOLERANCE_M)
        if arrived:
            self.call_magnet(False)
            if self.pick_attempt_count < MAX_PICK_ATTEMPTS:
                self.change_state(self.STATE_APPROACH_PRE_PICK, now)
            else:
                self.give_up_current(now)

    def state_contact_hold(self, now):
        '''
        Description:    Hold Still Post Contact.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if (now - self.state_enter_time) >= STANDSTILL_HOLD_S:
            self.change_state(self.STATE_CONTACT_LIFT, now)

    def state_contact_lift(self, now):
        '''
        Description:    Small Test Lift Post Contact (0.05m at 0.05 m/s).

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        ore_x, ore_y, _ = self.current_ore[2]
        target_test_lift = (ore_x, ore_y, self.contact_z + POST_CONTACT_LIFT_M)
        arrived = self.navigate_cartesian(target_test_lift, POST_CONTACT_LIFT_SPEED_MPS, now,
                                          tol=POSITION_TOLERANCE_M, gain=KP_TRANSIT)
        if arrived:
            self.publish_zero_twist()
            self.change_state(self.STATE_CONFIRM_GRIP, now)

    def state_confirm_grip(self, now):
        '''
        Description:    Wait 0.3s & Confirm Grip via Mean Force.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if (now - self.state_enter_time) >= POST_CONTACT_WAIT_S:
            forces_in_window = [sample[1] for sample in self.recent_forces if (now - sample[0]) <= POST_CONTACT_WAIT_S]
            mean_force = (sum(forces_in_window) / len(forces_in_window) if forces_in_window
                          else self.force_magnitude)
            held_force = PACKAGE_HELD_FORCE_N if self.is_package() else HELD_CONFIRM_FORCE_N
            if mean_force >= held_force:
                self.get_logger().info(
                    f'SUCCESSFUL PICK: {self.current_ore[0]} confirmed held (mean force={mean_force:.2f}N).'
                )
                picked_msg = String()
                picked_msg.data = self.current_ore[0]
                self.picked_pub.publish(picked_msg)
                self.ore_held = True
                # The package hangs clear of the deck: the eBot may drive home now
                if self.is_package() and not self.lift_reported:
                    self.lifted_pub.publish(Bool(data=True))
                    self.lift_reported = True
                    self.get_logger().info('Package off the eBot deck: /package_lifted sent')
                self.change_state(self.STATE_LIFT_AFTER_PICK, now)
            else:
                self.pick_attempt_count += 1
                self.get_logger().warn(
                    f'Pick failed for {self.current_ore[0]} (mean force={mean_force:.2f}N < {held_force}N). '
                    f'Attempt {self.pick_attempt_count}/{MAX_PICK_ATTEMPTS}'
                )
                self.call_magnet(False)
                if self.pick_attempt_count < MAX_PICK_ATTEMPTS:
                    self.change_state(self.STATE_APPROACH_PRE_PICK, now)
                else:
                    self.give_up_current(now)

    def state_lift_after_pick(self, now):
        '''
        Description:    Full Lift to Transit Height.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        ore_x, ore_y, _ = self.current_ore[2]
        target_lift = (ore_x, ore_y, self.pick_heights()[1])
        arrived = self.navigate_cartesian(target_lift, CRUISE_SPEED_MPS, now,
                                          tol=POSITION_TOLERANCE_M, through=True)
        if arrived:
            self.side_passed = False
            self.change_state(self.STATE_TRANSIT_HOME_WITH_ORE, now)

    def state_transit_home_with_ore(self, now):
        '''
        Description:    Transit Home Waypoint with Ore.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        arrived = self.via_side(SAFE_HOME_POSE, now)
        if arrived:
            self.change_state(self.STATE_APPROACH_PRE_DROP, now)

    def state_approach_pre_drop(self, now):
        '''
        Description:    Move to Pre-Drop Pose Above the Container (or the crusher hopper).

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        drop_x, drop_y, pre_z, drop_z, speed, is_far = self.drop_plan()
        self.target_orientation = self.tilted_orientation(drop_x, drop_y)
        reach = math.hypot(self.tcp_pose.position.x, self.tcp_pose.position.y)
        if is_far and reach < FAR_DROP_SLOW_REACH_M:
            speed = CRUISE_SPEED_MPS                  # slow down only near full reach
        if not is_far:
            # The release height keeps the item above the rim, so it can be reached directly
            if self.navigate_cartesian((drop_x, drop_y, drop_z), speed, now, gain=KP_TRANSIT):
                self.start_release(now)
            return
        arrived = self.navigate_cartesian((drop_x, drop_y, pre_z), speed, now, tol=PRE_DROP_TOL_M,
                                          gain=KP_TRANSIT)
        if arrived:
            self.change_state(self.STATE_DESCENT_DROP, now)

    def state_descent_drop(self, now):
        '''
        Description:    Descend into the Container (or to the release height over the hopper).

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        drop_x, drop_y, _, drop_z, _, _ = self.drop_plan()
        self.target_orientation = self.tilted_orientation(drop_x, drop_y)
        target_drop = (drop_x, drop_y, drop_z)
        arrived = self.navigate_cartesian(target_drop, DROP_DESCENT_SPEED_MPS, now, gain=KP_TRANSIT)
        if arrived:
            self.start_release(now)

    def start_release(self, now):
        '''
        Description:    Stops at the release point and starts switching the magnet off.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        self.publish_zero_twist()
        self.magnet_retry_count = 0
        self.last_magnet_call_time = 0.0
        self.magnet_future = None
        self.release_retry_count = 0
        self.release_force_ok_start_time = None
        self.change_state(self.STATE_DISABLE_MAGNET, now)

    def state_disable_magnet(self, now):
        '''
        Description:    De-energize Magnet with Retry Logic.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
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
                    response = self.magnet_future.result()
                    if response.success:
                        self.magnet_future = None
                        self.release_force_ok_start_time = None
                        self.change_state(self.STATE_CONFIRM_RELEASE, now)
                        return
                    else:
                        self.magnet_future = None
                except Exception as error:
                    self.get_logger().error(f'Magnet OFF error: {error}')
                    self.magnet_future = None

    def state_confirm_release(self, now):
        '''
        Description:    Confirm Release via Force Threshold Check.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if self.force_magnitude < RELEASE_CONFIRM_FORCE_N:
            if self.release_force_ok_start_time is None:
                self.release_force_ok_start_time = now
            elif (now - self.release_force_ok_start_time) >= RELEASE_CONFIRM_WINDOW_S:
                tcp_str, joints_str = self.tcp_text(), self.joints_text()
                cx_d, cy_d = self.drop_plan()[:2]
                tilt_deg = math.degrees(min(
                    max(0.0, math.asin(min(1.0, max(0.0, math.hypot(cx_d, cy_d) - COMFORT_REACH_M) / TOOL_LEN_M))),
                    math.radians(MAX_DROP_TILT_DEG)))
                self.get_logger().info(
                    f'SUCCESSFUL DROP: {self.current_ore[0]} released into '
                    f'{"the ore crusher" if self.is_package() else "its container"} '
                    f'(force={self.force_magnitude:.2f}N).'
                )
                self.get_logger().info(
                    f'[DROP-JOINTS] Ore: {self.current_ore[1]}, tilt={tilt_deg:.2f}deg, '
                    f'TCP: {tcp_str}, Joints: [{joints_str}]'
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

    def state_retreat_to_pre_drop(self, now):
        '''
        Description:    Retreat Upward to Pre-Drop Height.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        drop_x, drop_y, pre_z, _, _, is_far = self.drop_plan()
        self.target_orientation = self.tilted_orientation(drop_x, drop_y)
        if is_far:
            arrived = self.navigate_cartesian(SAFE_HOME_POSE, CRUISE_SPEED_MPS, now, through=True)
            if arrived:
                self.target_orientation = self.vertical_orientation
                self.change_state(self.STATE_RETURN_HOME, now)
        else:
            arrived = self.navigate_cartesian((drop_x, drop_y, pre_z), CRUISE_SPEED_MPS, now, through=True)
            if arrived:
                self.target_orientation = self.vertical_orientation
                self.change_state(self.STATE_RETURN_HOME, now)

    def state_return_home(self, now):
        '''
        Description:    Return to Safe Home Waypoint & Choose the Next Item.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        arrived = self.navigate_cartesian(SAFE_HOME_POSE, CRUISE_SPEED_MPS, now, through=True)
        if arrived:
            self.next_target(now)

    def state_latch_recovery_unlock(self, now):
        '''
        Description:    Latch Recovery -- Unlock Service Call.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if self.unlock_future is None:
            self.call_unlock()
        else:
            if self.unlock_future.done():
                try:
                    response = self.unlock_future.result()
                    self.get_logger().info(f'Unlock protective stop returned: {response.message}')
                except Exception as error:
                    self.get_logger().error(f'Unlock service error: {error}')
                self.unlock_future = None
                self.change_state(self.STATE_LATCH_RECOVERY_CLEAR, now)

    def state_latch_recovery_clear(self, now):
        '''
        Description:    Latch Recovery -- Immediate Upward Motion (Clear Fault).

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
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
                        f'Held-ore retreat count for {ore_id}: '
                        f'{self.ore_retreat_counts[ore_id]}/{MAX_HELD_RETREATS_PER_ORE}'
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
                    f'Latch count {self.latch_count} >= {MAX_LATCHES_BEFORE_RETREAT}. '
                    f'Initiating joint-space retreat to unfold targets...'
                )
                if self.ore_held is False and (self.current_ore is None
                                               or self.current_ore[0] not in self.done_ores):
                    self.call_magnet(False)
                    if self.current_ore is not None:
                        self.pick_attempt_count += 1
                        self.ore_queue.insert(0, self.current_ore)
                        self.current_ore = None
                self.retreat_resume_state = (self.STATE_TRANSIT_HOME_WITH_ORE if self.ore_held
                                             else self.STATE_MOVE_HOME)
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

    def state_all_completed(self, now):
        '''
        Description:    All Tasks Completed.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        self.publish_zero_twist()

    def state_wait_for_package(self, now):
        '''
        Description:    Every ore sorted: hold at the safe home pose until the package is ready.

        Args:
            now             (float):    current time in seconds

        Returns:
        '''
        if self.leave_ebot_side(now):
            self.navigate_cartesian(SAFE_HOME_POSE, CRUISE_SPEED_MPS, now)
        if self.package_ready():
            self.next_target(now)
        elif now - self.last_log_time > 5.0:
            self.get_logger().info('All ores sorted: waiting for the eBot and the ore package')
            self.last_log_time = now


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    arm_manipulation class to perform its task
    '''
    rclpy.init()
    node = arm_manipulation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_zero_twist()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
