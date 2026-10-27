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
# Filename:         task2B_planning.py
# Functions:
#                   wrap_angle, yaw_from_quat, snap_yaw, polyline_length, build_clearance,
#                   build_traversable, astar, has_line_of_sight, shortcut_turn_safe,
#                   refine_bends, densify, main
# Nodes:            Publishing Topics  - [ /ebot_path ]
#                   Subscribing Topics - [ /map, /odom ]
#                   Service Clients    - [ /spawn_ore_package ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_srvs.srv import Trigger

import heapq                                                            # priority queue for A*
import time                                                             # wall clock for the spawn retries
import numpy as np                                                      # grid maths for the costmap


##################### TASK CONSTANTS #######################

# The map is published ONCE, as the world loads, and held for late subscribers.
map_topic = '/map'

# Your follower reads the path from here, with the same held-message QoS.
path_topic = '/ebot_path'

odom_topic = '/odom'
spawn_service = '/spawn_ore_package'

# Every pose in the path, and the path itself, is in this frame.
map_frame = 'map'

# The three points of the lap: (x, y) in metres and yaw in radians, in the map frame.
# At the ore drop pose and the arm pose either yaw, 0.0 or 3.14, is valid.
home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, 0.0)
arm_pose = (0.0, 1.9, 0.0)

# The lap, as (start, end) of each step, in the order they are driven.
steps = [
    (home_pose, ore_drop_pose),     # step 1: then request the ore package
    (ore_drop_pose, arm_pose),      # step 2
    (arm_pose, home_pose),          # step 3
]

# Yaws accepted at each pose. Home faces 0; the other two may face either way.
yaw_choices = {home_pose: [0.0], ore_drop_pose: [0.0, math.pi], arm_pose: [0.0, math.pi]}


##################### PLANNER TUNING #######################

# The eBot's footprint is its chassis box, 0.62 m x 0.49 m (the wheels sit inside it).
# Turning on the spot sweeps the circle that encloses the box: sqrt(0.31^2 + 0.245^2).
# The benchmark path clears every rock by exactly this circle.
robot_radius = 0.395

# Clearance the eBot's CENTRE needs from every rock, in metres:
#   - turn_clearance:  wherever the path bends, because the eBot turns on the spot there.
#   - drive_clearance: along the straight stretches between bends. Driving straight, only
#                      the box's half-width (0.245 m) plus tracking error has to fit; the
#                      worst sideways reach measured in a real run was 0.358 m.
# These two numbers are the task's trade-off: lower = shorter paths (more EP), but less
# room for tracking error. 0.41 m opens the narrow corridors the benchmark itself uses.
turn_margin = 0.16
turn_clearance = robot_radius + turn_margin
drive_clearance = 0.41

# If no path fits those, plan with the plain circle rule at these margins instead.
fallback_margins = (0.08, 0.05, 0.02)

# Optional extra preference for room beyond the hard clearances above: steps closer than
# soft_clearance cost up to (1 + soft_weight) times their length. 0 = shortest path that
# meets the hard clearances (tested safest-and-shortest across 17 rock layouts).
soft_clearance = 0.85
soft_weight = 0.0

endpoint_relax_radius = 0.7     # m round the start/end poses where the clearance is relaxed
max_stub_length = 0.6           # m: furthest a tight pose is left/entered straight before turning
endpoint_keep_fraction = 0.5    # near a pose, keep at least this fraction of the required clearance...
endpoint_slack = 0.03           # ...or the clearance the eBot already has there, minus this (m)
plan_resolution = 0.025         # m: /map (5 mm cells) is pooled to this before planning
clearance_cap = soft_clearance + 0.05   # m: clearance is computed exactly up to this distance
publish_spacing = 0.10          # m between published poses (does not change the length)
refine_passes = 12              # passes of bend tightening after the shortcut
refine_tol = 0.001              # m: stop tightening once no bend moves more than this

# When the follower counts as finished with a step.
start_tolerance = 0.3           # m: the eBot counts as standing at a step's start within this
arrive_dist = 0.20              # m (the evaluator allows 0.3)
arrive_yaw = 0.10               # rad (the evaluator allows 0.15)
still_lin = 0.02                # m/s
still_ang = 0.02                # rad/s
still_hold_s = 1.0              # s standing still before the step is declared done

spawn_retry_s = 1.0             # s between /spawn_ore_package attempts
spawn_max_attempts = 5


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


def yaw_from_quat(q):
    '''
    Description:    Yaw of a geometry_msgs Quaternion.

    Args:
        q               (Quaternion):   orientation to read

    Returns:
        yaw             (float):    rotation about z, in radians
    '''
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def snap_yaw(yaw, choices):
    '''
    Description:    The allowed yaw closest to 'yaw'.

    Args:
        yaw             (float):    heading in radians
        choices         (list):     allowed headings in radians

    Returns:
        snapped         (float):    the entry of 'choices' nearest to 'yaw'
    '''
    return min(choices, key=lambda c: abs(wrap_angle(yaw - c)))


def polyline_length(points):
    '''
    Description:    Length of a polyline in metres.

    Args:
        points          (list):     [(x, y), ...] vertices in order

    Returns:
        length          (float):    sum of the segment lengths, in metres
    '''
    return sum(math.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
               for seg_start, seg_end in zip(points, points[1:]))


##################### PLANNING UTILITIES #######################

def build_clearance(occ, res, cap):
    '''
    Description:    Distance (m) from every cell to the nearest rock cell's edge, exact up to
                    'cap'. Shifts the occupancy mask over a disc of offsets (numpy only).

    Args:
        occ (np.ndarray):   bool grid, True = rock, shape (height, width)
        res (float):        metres per cell
        cap (float):        largest distance computed

    Returns:
        np.ndarray:         float grid of clearances, 0 on rock cells
    '''
    rows, cols = occ.shape
    big = cap / res
    dist = np.full((rows, cols), big, dtype=np.float32)
    radius_cells = int(math.ceil(big))
    offsets = sorted((math.hypot(dx, dy), dy, dx) for dy in range(-radius_cells, radius_cells + 1)
                     for dx in range(-radius_cells, radius_cells + 1) if math.hypot(dx, dy) <= big)
    for offset_dist, dy, dx in offsets:
        ys0, ys1 = max(0, -dy), min(rows, rows - dy)
        xs0, xs1 = max(0, -dx), min(cols, cols - dx)
        if ys0 < ys1 and xs0 < xs1:
            view = dist[ys0:ys1, xs0:xs1]
            np.minimum(view, np.where(occ[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx], offset_dist, big), out=view)
    clr = dist * res - 0.5 * res                # centre-to-centre -> to the rock cell's edge
    clr[occ] = 0.0
    return np.maximum(clr, 0.0)


def build_traversable(clr, hard, endpoints, cell_xy, relax_radius):
    '''
    Description:    Cells the eBot's centre may occupy: clearance >= 'hard', or near a start/
                    end pose with at least the clearance the eBot already has standing there.

    Args:
        clr (np.ndarray):       clearance grid
        hard (float):           required clearance in metres
        endpoints (list):       [((x, y), clearance_there), ...]
        cell_xy (tuple):        (X, Y) grids of cell-centre coordinates
        relax_radius (float):   metres round each endpoint where the margin is relaxed

    Returns:
        np.ndarray:             bool grid, True = traversable
    '''
    trav = clr >= hard
    X, Y = cell_xy
    for (ex, ey), ec in endpoints:
        need = max(endpoint_keep_fraction * min(ec, hard), min(hard, ec) - endpoint_slack)
        trav |= ((X - ex) ** 2 + (Y - ey) ** 2 <= relax_radius ** 2) & (clr >= need)
    return trav


def astar(trav, cost_mul, start, goal):
    '''
    Description:    8-connected A* over traversable cells; each step costs its length times
                    the mean cost multiplier of the two cells. No squeezing between corners.

    Args:
        trav            (np.ndarray):   bool grid of traversable cells
        cost_mul        (np.ndarray):   float grid of step-cost multipliers (>= 1)
        start           (tuple):    starting cell as (row, col)
        goal            (tuple):    target cell as (row, col)

    Returns:
        cells           (list):     [(row, col), ...] from start to goal, or None if unreachable
    '''
    rows, cols = trav.shape
    diag = math.sqrt(2)
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)]
    goal_row, goal_col = goal
    best_cost = {start: 0.0}
    parent = {start: None}
    heap = [(math.hypot(start[0] - goal_row, start[1] - goal_col), 0.0, start)]
    closed = set()
    while heap:
        _, cost_so_far, current_cell = heapq.heappop(heap)
        if current_cell in closed:
            continue
        if current_cell == goal:
            out = []
            while current_cell is not None:
                out.append(current_cell)
                current_cell = parent[current_cell]
            return out[::-1]
        closed.add(current_cell)
        row, col = current_cell
        for dr, dc, step in moves:
            next_row, next_col = row + dr, col + dc
            if not (0 <= next_row < rows and 0 <= next_col < cols) or not trav[next_row, next_col]:
                continue
            if dr and dc and not (trav[row, next_col] and trav[next_row, col]):
                continue
            new_cost = cost_so_far + step * 0.5 * (cost_mul[row, col] + cost_mul[next_row, next_col])
            if new_cost < best_cost.get((next_row, next_col), float('inf')):
                best_cost[(next_row, next_col)] = new_cost
                parent[(next_row, next_col)] = current_cell
                heapq.heappush(heap, (new_cost + math.hypot(next_row - goal_row, next_col - goal_col),
                                      new_cost, (next_row, next_col)))
    return None


def has_line_of_sight(start_point, end_point, trav, to_cell, step):
    '''
    Description:    True when every sample along the segment lies on a traversable cell.

    Args:
        start_point     (tuple):    (x, y) start of the segment
        end_point       (tuple):    (x, y) end of the segment
        trav            (np.ndarray):   bool grid of traversable cells
        to_cell         (function): (x, y) -> (row, col), or None outside the map
        step            (float):    sampling distance in metres

    Returns:
        clear           (bool):     True if the whole segment is traversable
    '''
    sample_count = max(1, int(math.ceil(
        math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1]) / step)))
    for sample_index in range(sample_count + 1):
        cell = to_cell(start_point[0] + (end_point[0] - start_point[0]) * sample_index / sample_count,
                       start_point[1] + (end_point[1] - start_point[1]) * sample_index / sample_count)
        if cell is None or not trav[cell]:
            return False
    return True


def shortcut_turn_safe(points, turn_ok, visible):
    '''
    Description:    Pulls a jagged grid path taut: from each kept point, jump to the farthest
                    later point in sight. A kept point (a bend, where the eBot turns on the
                    spot) must be turn-safe; the stretches between bends only need to be
                    'visible' (drivable). Done from both ends; first and last points kept.

    Args:
        points (list):          [(x, y), ...]
        turn_ok (list):         bool per point: safe to turn on the spot there
        visible (function):     (a, b) -> bool, drivable straight line

    Returns:
        list:                   the shortened polyline, or None if it cannot be built
    '''
    def one_pass(pts, oks):
        '''
        Description:    One greedy pass: from each kept point, jump to the farthest later point
                        that is in sight and safe to turn at.

        Args:
            pts             (list):     [(x, y), ...] points to shorten
            oks             (list):     bool per point: safe to turn on the spot there

        Returns:
            result          (tuple):    (kept points, their turn-safe flags), or None if a gap is not drivable
        '''
        out, keep, i = [pts[0]], [oks[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not (oks[j] and visible(pts[i], pts[j])):
                j -= 1
            if not visible(pts[i], pts[j]):
                return None
            out.append(pts[j])
            keep.append(oks[j])
            i = j
        return out, keep

    first = one_pass(points, turn_ok)
    if first is None:
        return None
    second = one_pass(first[0][::-1], first[1][::-1])
    return second[0][::-1] if second else first[0]


def refine_bends(points, visible, turn_ok, passes, tol):
    '''
    Description:    Tightens a taut path further. The shortcut keeps every bend on a grid-cell
                    centre, a little wider round its rock than needed. Each bend is slid
                    towards the straight line between its neighbours, as far as both legs stay
                    drivable and the eBot can still turn there; a bend that reaches the line is
                    dropped. Clearance rules are unchanged, only the length shrinks. The first
                    and last points never move.

    Args:
        points (list):          [(x, y), ...] from shortcut_turn_safe()
        visible (function):     (a, b) -> bool, drivable straight line
        turn_ok (function):     (x, y) -> bool, safe to turn on the spot there
        passes (int):           most passes over all bends
        tol (float):            stop once no bend moves more than this (m)

    Returns:
        list:                   the tightened polyline
    '''
    pts = list(points)
    for _ in range(passes):
        moved = 0.0
        i = 1
        while i < len(pts) - 1:
            (ax, ay), (vx, vy), (bx, by) = pts[i - 1], pts[i], pts[i + 1]
            if visible(pts[i - 1], pts[i + 1]):              # bend no longer needed
                moved = max(moved, math.hypot(vx - ax, vy - ay))
                del pts[i]
                continue
            # foot of the perpendicular from the bend onto the line through its neighbours
            dx, dy = bx - ax, by - ay
            foot_t = max(0.0, min(1.0, ((vx - ax) * dx + (vy - ay) * dy) / (dx * dx + dy * dy)))
            fx, fy = ax + foot_t * dx, ay + foot_t * dy
            lo, hi = 0.0, 1.0                                 # bisect the furthest safe fraction
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                candidate = (vx + mid * (fx - vx), vy + mid * (fy - vy))
                if turn_ok(*candidate) and visible(pts[i - 1], candidate) and visible(candidate, pts[i + 1]):
                    lo = mid
                else:
                    hi = mid
            candidate = (vx + lo * (fx - vx), vy + lo * (fy - vy))
            moved = max(moved, math.hypot(candidate[0] - vx, candidate[1] - vy))
            pts[i] = candidate
            i += 1
        if moved < tol:
            break
    return pts


def densify(points, spacing):
    '''
    Description:    Inserts evenly spaced points along every segment (length unchanged).

    Args:
        points          (list):     [(x, y), ...] vertices of the path
        spacing         (float):    largest gap between published points, in metres

    Returns:
        dense           (list):     [(x, y), ...] the same path with extra points
    '''
    out = [points[0]]
    for seg_start, seg_end in zip(points, points[1:]):
        piece_count = max(1, int(math.ceil(math.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]) / spacing)))
        out += [(seg_start[0] + (seg_end[0] - seg_start[0]) * piece_index / piece_count,
                 seg_start[1] + (seg_end[1] - seg_start[1]) * piece_index / piece_count)
                for piece_index in range(1, piece_count + 1)]
    return out


##################### CLASS DEFINITION #######################

class ebot_planner(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to read the map, plan a path for each
                    step of the lap, publish it on /ebot_path, and request the ore package
                    at the end of step 1.
    '''

    # State machine definitions
    STATE_WAIT_FOR_DATA = 0
    STATE_PLAN = 1
    STATE_DRIVING = 2
    STATE_REQUEST_PACKAGE = 3
    STATE_COMPLETED = 4

    def __init__(self):
        '''
        Description:    Initialization of class ebot_planner
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(                                                               # registering node
            'ebot_planner_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # A held message: kept by the publisher and handed to anyone who subscribes later.
        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############

        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self.mapcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)

        ############ Topic PUBLISHERS ############

        self.path_pub = self.create_publisher(Path, path_topic, latched)                # your follower drives what you publish here

        ############ Service CLIENTS ############

        self.spawn_client = self.create_client(Trigger, spawn_service)                  # drops the ore package onto the eBot

        ############ Constructor VARIABLES/OBJECTS ############

        plan_rate = 0.1                                                                 # rate of time to run one planning cycle (seconds)
        self.timer = self.create_timer(plan_rate, self.process_planning)                # creating a timer based function which gets called on every 0.1 seconds (as defined by 'plan_rate' variable)

        self.grid = None                                                                # the map (from mapcb())
        self.odom = None                                                                # where the base is (from odomcb())
        self.step = 0                                                                   # which step of the lap is next (index into 'steps')

        ############ ADD YOUR CODE HERE ############

        # Planning grid (filled by mapcb())
        self.occ = None             # bool grid of rock cells, at plan_resolution
        self.clr = None             # clearance of every cell, in metres
        self.res = None             # metres per planning cell
        self.origin = None          # (x, y) of the map's corner
        self.extent = None          # (x, y) of the map's far corner
        self.cell_xy = None         # (X, Y) grids of cell-centre coordinates

        # Base state (filled by odomcb())
        self.x = self.y = self.yaw = 0.0
        self.v = self.w = 0.0

        # Step state
        self.state = self.STATE_WAIT_FOR_DATA
        self.current_end = None                 # (x, y, yaw) the follower is heading for
        self.current_end_pose = None            # which of the three poses that is
        self.still_since = None
        self.spawn_pending = False
        self.spawn_attempts = 0
        self.last_spawn_try = None
        self.lengths = []                       # length of each published path

        ############################################


    def mapcb(self, data):
        '''
        Description:    Callback function for the map topic.
                        Use this function to receive the occupancy grid of the arena.

        Args:
            data (OccupancyGrid):    The arena, one value per cell

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        if self.grid is not None:
            return
        t0 = time.monotonic()
        info = data.info
        self.origin = (info.origin.position.x, info.origin.position.y)
        self.extent = (self.origin[0] + info.width * info.resolution,
                       self.origin[1] + info.height * info.resolution)
        self.get_logger().info(
            f'Map: {info.width} x {info.height} cells at {info.resolution:.4f} m, origin '
            f'({self.origin[0]:.2f}, {self.origin[1]:.2f}), extent to ({self.extent[0]:.2f}, {self.extent[1]:.2f})')

        # Pool the fine map into planning cells: a cell is a rock if ANY map cell in it is,
        # so rocks only ever grow. Beyond the map edge the padding is free, but to_cell()
        # rejects it, so no path leaves the map.
        fine = np.asarray(data.data, dtype=np.int16).reshape(info.height, info.width) != 0
        pool_factor = max(1, int(round(plan_resolution / info.resolution)))
        rows, cols = -(-info.height // pool_factor), -(-info.width // pool_factor)
        padded = np.zeros((rows * pool_factor, cols * pool_factor), dtype=bool)
        padded[:info.height, :info.width] = fine
        self.occ = padded.reshape(rows, pool_factor, cols, pool_factor).any(axis=(1, 3))
        self.res = info.resolution * pool_factor

        self.clr = build_clearance(self.occ, self.res, clearance_cap)
        self.cell_xy = np.meshgrid(self.origin[0] + (np.arange(cols) + 0.5) * self.res,
                                   self.origin[1] + (np.arange(rows) + 0.5) * self.res)
        self.grid = data
        self.get_logger().info(
            f'Planning grid: {cols} x {rows} cells at {self.res:.3f} m (x{pool_factor}), '
            f'{int(self.occ.sum())} rock cells, ready in {time.monotonic() - t0:.2f} s')

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
        self.v = data.twist.twist.linear.x
        self.w = data.twist.twist.angular.z
        self.odom = data

        ############################################


    def to_cell(self, x, y):
        '''
        Description:    Map coordinates to (row, col) of the planning grid.

        Args:
            x               (float):    x in the map frame, in metres
            y               (float):    y in the map frame, in metres

        Returns:
            cell            (tuple):    (row, col), or None outside the map
        '''
        if not (self.origin[0] <= x < self.extent[0] and self.origin[1] <= y < self.extent[1]):
            return None
        row = int(math.floor((y - self.origin[1]) / self.res))
        col = int(math.floor((x - self.origin[0]) / self.res))
        if 0 <= row < self.occ.shape[0] and 0 <= col < self.occ.shape[1]:
            return (row, col)
        return None


    def to_xy(self, cell):
        '''
        Description:    Centre of a planning cell in map coordinates.

        Args:
            cell            (tuple):    (row, col) of the planning grid

        Returns:
            point           (tuple):    (x, y) in the map frame, in metres
        '''
        return (self.origin[0] + (cell[1] + 0.5) * self.res, self.origin[1] + (cell[0] + 0.5) * self.res)


    def is_still(self):
        '''
        Description:    True when the base has no linear or angular velocity.

        Args:

        Returns:
            still           (bool):     True if both speeds are below the stillness thresholds
        '''
        return abs(self.v) < still_lin and abs(self.w) < still_ang


    def plan_path(self, start, end):
        '''
        Description:    Plans a path from 'start' to 'end' that stays clear of every rock.

        Args:
            start (tuple):    (x, y, yaw) where the step begins, in the map frame
            end (tuple):      (x, y, yaw) where the step ends, in the map frame

        Returns:
            list:             [(x, y), ...] from 'start' to 'end', or None if no path was found
        '''

        points = None

        ############ ADD YOUR CODE HERE ############

        start_cell, goal_cell = self.to_cell(start[0], start[1]), self.to_cell(end[0], end[1])
        if start_cell is None or goal_cell is None:
            self.get_logger().error('Start or end pose lies outside the map')
            return None

        start_xy, end_xy = (start[0], start[1]), (end[0], end[1])
        endpoints = [(start_xy, float(self.clr[start_cell])), (end_xy, float(self.clr[goal_cell]))]

        # Rules tried in order: (drive clearance, turn clearance). The first is the split
        # rule; the rest are the plain circle rule (same clearance everywhere).
        rules = [(drive_clearance, turn_clearance)] + [(robot_radius + margin,) * 2 for margin in fallback_margins]
        for rule, (drive, turn) in enumerate(rules):
            # Drivable may be relaxed round the poses (the eBot is already standing there);
            # turnable never is, so the eBot only ever turns where it really has room.
            drivable = build_traversable(self.clr, drive, endpoints, self.cell_xy, endpoint_relax_radius)
            turnable = self.clr >= turn
            if not (drivable[start_cell] and drivable[goal_cell]):
                continue
            penalty = np.clip((soft_clearance - self.clr) / max(soft_clearance - drive, 1e-3), 0.0, 1.0)
            cost = 1.0 + soft_weight * penalty           # steps near rocks cost up to (1 + soft_weight) x

            def visible(p, q):
                '''
                Description:    True when the straight line p-q stays on drivable cells.

                Args:
                    p               (tuple):    (x, y) start of the line
                    q               (tuple):    (x, y) end of the line

                Returns:
                    clear           (bool):     True if the eBot can drive straight from p to q
                '''
                return has_line_of_sight(p, q, drivable, self.to_cell, self.res * 0.25)

            # A tight pose is left and entered along its heading (the x axis: every allowed
            # yaw is 0 or pi), forwards or backwards, to the first spot with room to turn.
            best = None
            for start_stub in self.stubs(start_xy, start[2], visible, turnable):
                for end_stub in self.stubs(end_xy, end[2], visible, turnable):
                    middle = self.search(start_stub, end_stub, drivable, turnable, cost, visible)
                    if middle is None:
                        continue
                    pts = (([start_xy] if start_stub != start_xy else []) + middle
                           + ([end_xy] if end_stub != end_xy else []))
                    if best is None or polyline_length(pts) < polyline_length(best):
                        best = pts
            if best is None:
                continue
            points = best
            if rule > 0:
                self.get_logger().warn(f'Split rule found no path; planned with the circle rule at '
                                       f'{drive:.3f} m clearance')
            break

        ############################################

        return points


    def stubs(self, pose, yaw, visible, turnable):
        '''
        Description:    Where the eBot may turn near a pose. The pose itself when it has room
                        to turn; otherwise, for each direction along 'yaw' (forwards and
                        backwards), the nearest point with room that a straight line reaches.

        Args:
            pose (tuple):           (x, y)
            yaw (float):            heading the eBot stands at (or must finish at) there
            visible (function):     (a, b) -> bool, drivable straight line
            turnable (np.ndarray):  bool grid, room to turn on the spot

        Returns:
            list:                   candidate (x, y) points; [pose] if nothing better is found
        '''
        cell = self.to_cell(*pose)
        if cell is not None and turnable[cell]:
            return [pose]
        found = []
        for heading in (yaw, yaw + math.pi):
            for distance in np.arange(self.res, max_stub_length + 1e-9, self.res):
                candidate = (pose[0] + distance * math.cos(heading), pose[1] + distance * math.sin(heading))
                candidate_cell = self.to_cell(*candidate)
                if candidate_cell is None or not visible(pose, candidate):
                    break
                if turnable[candidate_cell]:
                    found.append(candidate)
                    break
        return found or [pose]


    def search(self, start_xy, end_xy, drivable, turnable, cost, visible):
        '''
        Description:    A* between two points, then pulled taut so that every bend is turn-safe.

        Args:
            start_xy        (tuple):    (x, y) start point
            end_xy          (tuple):    (x, y) end point
            drivable        (np.ndarray):   bool grid, room to drive straight
            turnable        (np.ndarray):   bool grid, room to turn on the spot
            cost            (np.ndarray):   float grid of step-cost multipliers (>= 1)
            visible         (function): (p, q) -> bool, drivable straight line

        Returns:
            points          (list):     [(x, y), ...] from start_xy to end_xy exactly, or None
        '''
        start_cell, end_cell = self.to_cell(*start_xy), self.to_cell(*end_xy)
        if start_cell is None or end_cell is None or not (drivable[start_cell] and drivable[end_cell]):
            return None
        cells = astar(drivable, cost, start_cell, end_cell)
        if cells is None:
            return None
        raw = [start_xy] + [self.to_xy(grid_cell) for grid_cell in cells[1:-1]] + [end_xy]
        turn_ok = [True] + [bool(turnable[grid_cell]) for grid_cell in cells[1:-1]] + [True]
        taut = shortcut_turn_safe(raw, turn_ok, visible)
        if taut is None:
            return None

        def can_turn(x, y):
            '''
            Description:    True when the eBot has room to turn on the spot at (x, y).

            Args:
                x               (float):    x in the map frame, in metres
                y               (float):    y in the map frame, in metres

            Returns:
                ok              (bool):     True if that cell is turnable
            '''
            cell = self.to_cell(x, y)
            return cell is not None and bool(turnable[cell])

        return refine_bends(taut, visible, can_turn, refine_passes, refine_tol)


    def publish_path(self, points, start, end):
        '''
        Description:    Publishes 'points' on /ebot_path as a nav_msgs/Path.

        Args:
            points (list):    [(x, y), ...] as returned by plan_path()
            start (tuple):    (x, y, yaw) where the step begins
            end (tuple):      (x, y, yaw) where the step ends

        Returns:
        '''

        path = Path()
        path.header.frame_id = map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        ############ ADD YOUR CODE HERE ############

        dense = densify(points, publish_spacing)
        dense[0], dense[-1] = (start[0], start[1]), (end[0], end[1])
        for i, (x, y) in enumerate(dense):
            if i == 0:
                yaw = start[2]
            elif i == len(dense) - 1:
                yaw = end[2]
            else:                                                            # face along the path
                yaw = math.atan2(dense[i + 1][1] - y, dense[i + 1][0] - x)
            pose = PoseStamped()
            pose.header.frame_id = map_frame
            pose.header.stamp = path.header.stamp
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            path.poses.append(pose)
        self.path_pub.publish(path)

        ############################################


    def request_ore_package(self):
        '''
        Description:    Calls /spawn_ore_package once the eBot stands still on the ore drop pose.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        if not self.spawn_client.service_is_ready():
            self.get_logger().warn('Waiting for /spawn_ore_package ...', throttle_duration_sec=2.0)
            return
        self.spawn_pending = True
        self.spawn_attempts += 1
        future = self.spawn_client.call_async(Trigger.Request())             # never block on the reply
        future.add_done_callback(self.spawn_done)

        ############################################


    def spawn_done(self, future):
        '''
        Description:    Handles the /spawn_ore_package reply: moves on to step 2 on success.

        Args:
            future          (Future):   the completed Trigger call

        Returns:
        '''
        self.spawn_pending = False
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f'Ore package call failed: {exc}')
            return
        self.get_logger().info(f'Ore package: success={result.success} "{result.message}"')
        if result.success:
            self.next_step()


    def next_step(self):
        '''
        Description:    Moves on to the next step of the lap, or finishes after step 3.

        Args:

        Returns:
        '''
        self.step += 1
        self.still_since = None
        if self.step < len(steps):
            self.state = self.STATE_PLAN
            return
        self.state = self.STATE_COMPLETED
        self.get_logger().info(
            f'*** LAP COMPLETE *** path lengths {[round(length, 2) for length in self.lengths]} m, '
            f'total {sum(self.lengths):.2f} m')


    def process_planning(self):
        '''
        Description:    Timer function used to plan, publish and request, step by step.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            now = self.get_clock().now().nanoseconds * 1e-9

            # ------------------------------------------------------------------
            # STATE 0: Wait for the map and odometry
            # ------------------------------------------------------------------
            if self.state == self.STATE_WAIT_FOR_DATA:
                if self.grid is not None and self.odom is not None:
                    self.state = self.STATE_PLAN

            # ------------------------------------------------------------------
            # STATE 1: Plan and publish, while the eBot stands still at the step's start
            # ------------------------------------------------------------------
            elif self.state == self.STATE_PLAN:
                start_pose, end_pose = steps[self.step]
                at_start = math.hypot(self.x - start_pose[0], self.y - start_pose[1]) <= start_tolerance
                if not at_start or not self.is_still():
                    self.get_logger().info(
                        f'Step {self.step + 1}: waiting for the eBot to stand still at the start '
                        f'(at {self.x:.2f}, {self.y:.2f}, v={self.v:.3f}, w={self.w:.3f})',
                        throttle_duration_sec=2.0)
                    return
                start_yaw = snap_yaw(self.yaw, yaw_choices[start_pose])        # the heading it really has
                start = (start_pose[0], start_pose[1], start_yaw)
                points = self.plan_path(start, end_pose)
                if points is None:
                    self.get_logger().error(f'No path found for step {self.step + 1}; retrying',
                                            throttle_duration_sec=2.0)
                    return
                (ax, ay), (bx, by) = points[-2], points[-1]                  # end yaw: match the arrival
                end = (end_pose[0], end_pose[1], snap_yaw(math.atan2(by - ay, bx - ax), yaw_choices[end_pose]))
                self.publish_path(points, start, end)
                self.lengths.append(polyline_length(points))
                self.current_end = end
                self.current_end_pose = end_pose
                self.state = self.STATE_DRIVING
                self.get_logger().info(f'Step {self.step + 1}: published path, {len(points)} vertices, '
                                       f'{self.lengths[-1]:.2f} m, end yaw {end[2]:.2f}')

            # ------------------------------------------------------------------
            # STATE 2: Wait for the follower to stop on the step's end pose
            # ------------------------------------------------------------------
            elif self.state == self.STATE_DRIVING:
                ex, ey, _ = self.current_end
                facing = min(abs(wrap_angle(self.yaw - choice)) for choice in yaw_choices[self.current_end_pose])
                arrived = (math.hypot(self.x - ex, self.y - ey) < arrive_dist and
                           facing < arrive_yaw and self.is_still())       # either allowed yaw counts
                if not arrived:
                    self.still_since = None
                    return
                if self.still_since is None:
                    self.still_since = now
                    return
                if now - self.still_since < still_hold_s:
                    return
                self.get_logger().info(f'Step {self.step + 1} finished')
                if self.step == 0:
                    self.state = self.STATE_REQUEST_PACKAGE
                    self.last_spawn_try = None
                else:
                    self.next_step()

            # ------------------------------------------------------------------
            # STATE 3: Request the ore package at the end of step 1
            # ------------------------------------------------------------------
            elif self.state == self.STATE_REQUEST_PACKAGE:
                if self.spawn_pending:
                    return
                if self.last_spawn_try is not None and now - self.last_spawn_try < spawn_retry_s:
                    return
                if self.spawn_attempts >= spawn_max_attempts:
                    self.get_logger().error(f'Ore package refused {spawn_max_attempts} times; continuing the lap')
                    self.next_step()
                    return
                self.last_spawn_try = now
                self.request_ore_package()

            # STATE 4: Completed - nothing more to do

        except Exception as exc:                                             # never let one cycle kill the node
            self.get_logger().error(f'process_planning: {exc}')

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_planner class to perform its task
    '''

    rclpy.init(args=sys.argv)                                           # initialisation

    node = rclpy.create_node('ebot_planner_process')                    # creating ROS node

    node.get_logger().info('Node created: eBot planner process')        # logging information

    ebot_planner_class = ebot_planner()                                 # creating a new object for class 'ebot_planner'

    rclpy.spin(ebot_planner_class)                                      # spining on the object to make it alive in ROS 2 DDS

    ebot_planner_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                    # shutdown process


if __name__ == '__main__':

    main()
