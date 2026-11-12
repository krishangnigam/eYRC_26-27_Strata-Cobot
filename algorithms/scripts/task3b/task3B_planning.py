#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 3B of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task3B_planning.py
# Functions:
#                   wrap_angle, yaw_from_quat, snap_yaw, polyline_length, build_clearance,
#                   build_traversable, astar, has_line_of_sight, shortest_turn_safe,
#                   shortcut_turn_safe, refine_bends, merge_short_segments, densify, main
# Nodes:            Publishing Topics  - [ /ebot_path, /ebot_at_arm ]
#                   Subscribing Topics - [ /map, /odom, /package_lifted ]
#                   Service Clients    - [ /spawn_ore_package ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from collections import deque                                           # recent odometry speeds
import heapq                                                            # priority queue for A*
import time                                                             # measures how long the map takes to process
import numpy as np                                                      # grid maths for the costmap


##################### TASK CONSTANTS #######################

# The map is published ONCE, as the world loads, and held for late subscribers.
map_topic = '/map'

# Your follower reads the path from here, with the same held-message QoS.
path_topic = '/ebot_path'

odom_topic = '/odom'
spawn_service = '/spawn_ore_package'

# Handover with the arm (topics of our own, both latched):
#   /ebot_at_arm     published here once the eBot stands still beside the arm with the package
#   /package_lifted  published by the manipulation node once the package is off the deck
ebot_at_arm_topic = '/ebot_at_arm'
package_lifted_topic = '/package_lifted'

# Every pose in the path, and the path itself, is in this frame.
map_frame = 'map'

# The three points of the lap: (x, y) in metres and yaw in radians, in the map frame.
# At the ore drop pose and the arm stop pose either yaw, 0.0 or 3.14, is valid.
home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, 0.0)

# Stop beside the arm (the arm's base_link is at map (-0.597, 1.865)). Here the package on
# the top deck is about 0.6 m in front of the arm base: within its reach, in the camera's
# view, and clear of the west wall (x = -0.462), which the eBot's rear stays 0.15 m from.
arm_stop_pose = (0.0, 1.9, 0.0)

# The lap, as (start, end) of each leg, in the order they are driven.
steps = [
    (home_pose, ore_drop_pose),     # leg 1: then request the ore package
    (ore_drop_pose, arm_stop_pose), # leg 2: then wait while the arm takes the package
    (arm_stop_pose, home_pose),     # leg 3
]
handover_step = 1                   # index of the leg that ends beside the arm

# Yaws accepted at each pose. Home faces 0; the other two may face either way.
yaw_choices = {home_pose: [0.0], ore_drop_pose: [0.0, math.pi], arm_stop_pose: [0.0, math.pi]}


##################### PLANNER TUNING #######################

# The eBot's footprint is its chassis box, 0.62 m x 0.49 m (the wheels sit inside it).
# Turning on the spot sweeps the circle that encloses the box: sqrt(0.31^2 + 0.245^2).
# The benchmark path clears every rock by exactly this circle.
robot_radius = 0.395

# Clearance the eBot's CENTRE needs from every rock, in metres:
#   - turn_clearance:   at every bend, because the eBot turns on the spot there and its
#                       front corners sweep a 0.478 m circle round the pivot.
#   - drive_clearances: along the straight stretches between bends, tried widest first.
#                       Task 3 does not score the path's length, so a wide berth comes
#                       first and the narrow corridors are the last resort.
turn_margin = 0.16
turn_clearance = robot_radius + turn_margin
drive_clearances = (0.56, 0.50, 0.45, 0.41)

# Task 3B: a rock may stand a little away from where /map shows it. A bend is where the
# eBot has the least room to spare (its corners swing out as it turns), so the room wanted
# at every bend is first widened by the largest of these margins; if no path fits, the
# next one is tried. The arm is what ends the run, not the eBot, so a longer and safer
# path costs nothing.
rock_shift_margins = (0.12, 0.08, 0.04, 0.0)

# Cells with no room to turn cost this much extra per metre in the search, so the path
# rounds corners out in the open and crosses a tight stretch by the shortest way.
tight_weight = 2.0

# If no path fits those, plan with the plain circle rule at these margins instead.
fallback_margins = (0.08, 0.05, 0.02)

# Optional extra preference for room beyond the hard clearances above: steps closer than
# soft_clearance cost up to (1 + soft_weight) times their length.
soft_clearance = 0.85
soft_weight = 0.0

endpoint_relax_radius = 0.7     # m round the start/end poses where the clearance is relaxed
max_stub_length = 0.6           # m: furthest a tight pose is left/entered straight before turning
stub_margin = 0.06              # m of extra room wanted where that first or last turn is made
endpoint_keep_fraction = 0.5    # near a pose, keep at least this fraction of the required clearance...
endpoint_slack = 0.03           # ...or the clearance the eBot already has there, minus this (m)
plan_resolution = 0.025         # m: /map (5 mm cells) is pooled to this before planning
clearance_cap = soft_clearance + 0.05   # m: clearance is computed exactly up to this distance
publish_spacing = 0.10          # m between published poses (does not change the length)
bend_stride = 1                 # one grid-path point in this many may become a bend
min_segment_length = 0.10       # m: a shorter segment between two bends is merged away
refine_passes = 12              # passes of bend tightening after the shortcut
refine_tol = 0.001              # m: stop tightening once no bend moves more than this

# When the follower counts as finished with a step.
start_tolerance = 0.3           # m: the eBot counts as standing at a step's start within this
arrive_dist = 0.20              # m (the evaluator allows 0.3)
arrive_yaw = 0.10               # rad (the evaluator allows 0.15)
still_lin = 0.02                # m/s
still_ang = 0.02                # rad/s
still_hold_s = 0.5              # s standing still before the step is declared done
still_window_s = 0.5            # s of odometry averaged, so sensor noise cannot hide a standstill
still_timeout_s = 3.0           # s on the pose after which it counts as standing, whatever the speeds read

spawn_retry_s = 0.5             # s between /spawn_ore_package attempts
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


def yaw_from_quat(quat):
    '''
    Description:    Yaw of a geometry_msgs Quaternion.

    Args:
        quat            (Quaternion):   orientation to read

    Returns:
        yaw             (float):    rotation about z, in radians
    '''
    return math.atan2(2.0 * (quat.w * quat.z + quat.x * quat.y), 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z))


def snap_yaw(yaw, choices):
    '''
    Description:    The allowed yaw closest to 'yaw'.

    Args:
        yaw             (float):    heading in radians
        choices         (list):     allowed headings in radians

    Returns:
        snapped         (float):    the entry of 'choices' nearest to 'yaw'
    '''
    return min(choices, key=lambda choice: abs(wrap_angle(yaw - choice)))


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

def build_clearance(occ, resolution, cap):
    '''
    Description:    Distance (m) from every cell to the nearest rock cell's edge, exact up to
                    'cap'. Shifts the occupancy mask over a disc of offsets (numpy only).

    Args:
        occ             (np.ndarray):   bool grid, True = rock, shape (height, width)
        resolution      (float):        metres per cell
        cap             (float):        largest distance computed

    Returns:
        clearance_grid  (np.ndarray):   float grid of clearances, 0 on rock cells
    '''
    rows, cols = occ.shape
    big = cap / resolution
    dist = np.full((rows, cols), big, dtype=np.float32)
    radius_cells = int(math.ceil(big))
    offsets = sorted((math.hypot(col_shift, row_shift), row_shift, col_shift)
                     for row_shift in range(-radius_cells, radius_cells + 1)
                     for col_shift in range(-radius_cells, radius_cells + 1) if math.hypot(col_shift, row_shift) <= big)
    for offset_dist, row_shift, col_shift in offsets:
        ys0, ys1 = max(0, -row_shift), min(rows, rows - row_shift)
        xs0, xs1 = max(0, -col_shift), min(cols, cols - col_shift)
        if ys0 < ys1 and xs0 < xs1:
            view = dist[ys0:ys1, xs0:xs1]
            np.minimum(view, np.where(occ[ys0 + row_shift:ys1 + row_shift, xs0 + col_shift:xs1 + col_shift],
                                      offset_dist, big), out=view)
    clr = dist * resolution - 0.5 * resolution  # centre-to-centre -> to the rock cell's edge
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
        traversable_grid    (np.ndarray):   bool grid, True = traversable
    '''
    trav = clr >= hard
    grid_x, grid_y = cell_xy
    for (end_x, end_y), end_clearance in endpoints:
        need = max(endpoint_keep_fraction * min(end_clearance, hard), min(hard, end_clearance) - endpoint_slack)
        trav |= ((grid_x - end_x) ** 2 + (grid_y - end_y) ** 2 <= relax_radius ** 2) & (clr >= need)
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
        for row_move, col_move, step in moves:
            next_row, next_col = row + row_move, col + col_move
            if not (0 <= next_row < rows and 0 <= next_col < cols) or not trav[next_row, next_col]:
                continue
            if row_move and col_move and not (trav[row, next_col] and trav[next_row, col]):
                continue
            new_cost = cost_so_far + step * 0.5 * (cost_mul[row, col] + cost_mul[next_row, next_col])
            if new_cost < best_cost.get((next_row, next_col), float('inf')):
                best_cost[(next_row, next_col)] = new_cost
                parent[(next_row, next_col)] = current_cell
                heapq.heappush(heap, (new_cost + math.hypot(next_row - goal_row, next_col - goal_col),
                                      new_cost, (next_row, next_col)))
    return None


def has_line_of_sight(start_point, end_point, trav, grid, step):
    '''
    Description:    True when every sample along the segment lies on a traversable cell.

    Args:
        start_point     (tuple):    (x, y) start of the segment
        end_point       (tuple):    (x, y) end of the segment
        trav            (np.ndarray):   bool grid of traversable cells
        grid            (tuple):    (origin x, origin y, extent x, extent y, metres per cell)
        step            (float):    sampling distance in metres

    Returns:
        clear           (bool):     True if the whole segment is traversable
    '''
    origin_x, origin_y, extent_x, extent_y, resolution = grid
    sample_count = max(1, int(math.ceil(
        math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1]) / step)))
    fraction = np.arange(sample_count + 1) / sample_count
    sample_x = start_point[0] + (end_point[0] - start_point[0]) * fraction
    sample_y = start_point[1] + (end_point[1] - start_point[1]) * fraction
    if (sample_x.min() < origin_x or sample_x.max() >= extent_x
            or sample_y.min() < origin_y or sample_y.max() >= extent_y):
        return False
    rows = np.floor((sample_y - origin_y) / resolution).astype(int)
    cols = np.floor((sample_x - origin_x) / resolution).astype(int)
    if rows.max() >= trav.shape[0] or cols.max() >= trav.shape[1]:
        return False
    return bool(trav[rows, cols].all())


def shortest_turn_safe(points, turn_ok, visible, stride):
    '''
    Description:    Shortest polyline along a jagged grid path whose every bend has room to turn
                    on the spot. The candidate bends are the turn-safe points, one in every
                    'stride'; two of them are joined when the straight line between them is
                    drivable. A narrow gap, where nothing is turn-safe, is crossed in a single
                    straight line, with the bends before and after it.

    Args:
        points          (list):     [(x, y), ...] the grid path
        turn_ok         (list):     bool per point: safe to turn on the spot there
        visible         (function): (a, b) -> bool, drivable straight line
        stride          (int):      one candidate bend is taken in every 'stride' points

    Returns:
        shortened_path  (list):     the polyline, or None if no such polyline exists
    '''
    last = len(points) - 1
    nodes = [0] + [i for i in range(stride, last, stride) if turn_ok[i]] + [last]
    best = {0: (0.0, None)}                     # node -> (length so far, node before it)
    for end_index in range(1, len(nodes)):
        tail = points[nodes[end_index]]
        options = sorted((best[start_index][0] + math.dist(points[nodes[start_index]], tail), start_index)
                         for start_index in range(end_index) if start_index in best)
        for length, start_index in options:     # shortest first: the first drivable one is the best
            if visible(points[nodes[start_index]], tail):
                best[end_index] = (length, start_index)
                break
    if len(nodes) - 1 not in best:
        return None
    path, node = [], len(nodes) - 1
    while node is not None:
        path.append(points[nodes[node]])
        node = best[node][1]
    return path[::-1]


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
        shortened_path  (list):     the shortened polyline, or None if it cannot be built
    '''
    def one_pass(points_to_shorten, turn_safe_flags):
        '''
        Description:    One greedy pass: from each kept point, jump to the farthest later point
                        that is in sight and safe to turn at.

        Args:
            points_to_shorten   (list):     [(x, y), ...] points to shorten
            turn_safe_flags     (list):     bool per point: safe to turn on the spot there

        Returns:
            result          (tuple):    (kept points, their turn-safe flags),
                                        or None if a gap is not drivable
        '''
        kept_points, kept_flags, i = [points_to_shorten[0]], [turn_safe_flags[0]], 0
        last = len(points_to_shorten) - 1
        while i < last:
            j = last
            while j > i + 1 and not (turn_safe_flags[j] and visible(points_to_shorten[i], points_to_shorten[j])):
                j -= 1
            if not visible(points_to_shorten[i], points_to_shorten[j]):
                return None
            kept_points.append(points_to_shorten[j])
            kept_flags.append(turn_safe_flags[j])
            i = j
        return kept_points, kept_flags

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
        refined_points  (list):     the tightened polyline
    '''
    pts = list(points)
    for _ in range(passes):
        moved = 0.0
        i = 1
        while i < len(pts) - 1:
            (prev_x, prev_y), (bend_x, bend_y), (next_x, next_y) = pts[i - 1], pts[i], pts[i + 1]
            if visible(pts[i - 1], pts[i + 1]):              # bend no longer needed
                moved = max(moved, math.hypot(bend_x - prev_x, bend_y - prev_y))
                del pts[i]
                continue
            # foot of the perpendicular from the bend onto the line through its neighbours
            chord_x, chord_y = next_x - prev_x, next_y - prev_y
            foot_t = max(0.0, min(1.0, ((bend_x - prev_x) * chord_x + (bend_y - prev_y) * chord_y)
                                  / (chord_x * chord_x + chord_y * chord_y)))
            foot_x, foot_y = prev_x + foot_t * chord_x, prev_y + foot_t * chord_y
            low, high = 0.0, 1.0                              # bisect the furthest safe fraction
            for _ in range(10):
                mid = 0.5 * (low + high)
                candidate = (bend_x + mid * (foot_x - bend_x), bend_y + mid * (foot_y - bend_y))
                if turn_ok(*candidate) and visible(pts[i - 1], candidate) and visible(candidate, pts[i + 1]):
                    low = mid
                else:
                    high = mid
            candidate = (bend_x + low * (foot_x - bend_x), bend_y + low * (foot_y - bend_y))
            moved = max(moved, math.hypot(candidate[0] - bend_x, candidate[1] - bend_y))
            pts[i] = candidate
            i += 1
        if moved < tol:
            break
    return pts


def merge_short_segments(points, visible, min_length):
    '''
    Description:    Removes a bend that sits closer than min_length to the next one, when the
                    path stays drivable without it. The first and last points are kept.

    Args:
        points          (list):     [(x, y), ...] vertices of the path
        visible         (function): (a, b) -> bool, drivable straight line
        min_length      (float):    shortest segment worth keeping, in metres

    Returns:
        merged_points   (list):     the path without the needless short segments
    '''
    pts = list(points)
    i = 0
    while i < len(pts) - 1:
        if math.dist(pts[i], pts[i + 1]) < min_length:
            if i + 2 < len(pts) and visible(pts[i], pts[i + 2]):
                del pts[i + 1]
                continue
            if i > 0 and visible(pts[i - 1], pts[i + 1]):
                del pts[i]
                i -= 1
                continue
        i += 1
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
        seg_length = math.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
        piece_count = max(1, int(math.ceil(seg_length / spacing)))
        out += [(seg_start[0] + (seg_end[0] - seg_start[0]) * piece_index / piece_count,
                 seg_start[1] + (seg_end[1] - seg_start[1]) * piece_index / piece_count)
                for piece_index in range(1, piece_count + 1)]
    return out


##################### CLASS DEFINITION #######################

class ebot_planner(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to read the map, plan a path for each
                    leg of the lap, publish it on /ebot_path, request the ore package at the
                    end of leg 1, and hand the package over to the arm at the end of leg 2.
    '''

    # State machine definitions
    STATE_WAIT_FOR_DATA = 0
    STATE_PLAN = 1
    STATE_DRIVING = 2
    STATE_REQUEST_PACKAGE = 3
    STATE_WAIT_HANDOVER = 4
    STATE_COMPLETED = 5

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
        self.lifted_sub = self.create_subscription(Bool, package_lifted_topic, self.liftedcb, latched)

        ############ Topic PUBLISHERS ############

        self.path_pub = self.create_publisher(Path, path_topic, latched)                # your follower drives what you publish here
        # tells the arm the package is beside it
        self.at_arm_pub = self.create_publisher(Bool, ebot_at_arm_topic, latched)

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
        self.linear_speed = self.turn_rate = 0.0
        self.recent_speeds = deque()            # (time, linear, angular) over the last still_window_s

        # Step state
        self.state = self.STATE_WAIT_FOR_DATA
        self.current_end = None                 # (x, y, yaw) the follower is heading for
        self.current_end_pose = None            # which of the three poses that is
        self.still_since = None
        self.near_since = None         # when the eBot came within tolerance of the pose it is wanted on
        self.spawn_pending = False
        self.spawn_attempts = 0
        self.last_spawn_try = None
        self.lengths = []                       # length of each published path
        self.package_lifted = False             # set by liftedcb() once the arm has the package

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
        start_time = time.monotonic()
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
            f'{int(self.occ.sum())} rock cells, ready in {time.monotonic() - start_time:.2f} s')

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
        self.linear_speed = data.twist.twist.linear.x
        self.turn_rate = data.twist.twist.angular.z
        self.odom = data

        # Keep the last still_window_s of speeds: their mean is what is_still() checks
        now = self.get_clock().now().nanoseconds * 1e-9
        self.recent_speeds.append((now, self.linear_speed, self.turn_rate))
        while self.recent_speeds and now - self.recent_speeds[0][0] > still_window_s:
            self.recent_speeds.popleft()

        ############################################


    def liftedcb(self, data):
        '''
        Description:    Callback function for /package_lifted. The manipulation node reports
                        that the package is off the eBot's deck, so the eBot may leave.

        Args:
            data            (Bool):     True once the package has been lifted

        Returns:
        '''
        if data.data and not self.package_lifted:
            self.package_lifted = True
            self.get_logger().info('Package lifted by the arm: the eBot may leave')


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
        Description:    True when the base has no linear or angular velocity, judged on the
                        mean of the last still_window_s of odometry, so that zero-mean sensor
                        noise on a standing eBot does not read as motion.

        Args:

        Returns:
            still           (bool):     True if both mean speeds are below the stillness thresholds
        '''
        if not self.recent_speeds:
            return abs(self.linear_speed) < still_lin and abs(self.turn_rate) < still_ang
        count = len(self.recent_speeds)
        mean_linear = sum(sample[1] for sample in self.recent_speeds) / count
        mean_angular = sum(sample[2] for sample in self.recent_speeds) / count
        return abs(mean_linear) < still_lin and abs(mean_angular) < still_ang


    def plan_path(self, start, end):
        '''
        Description:    Plans a path from 'start' to 'end' that stays clear of every rock.

        Args:
            start (tuple):    (x, y, yaw) where the step begins, in the map frame
            end (tuple):      (x, y, yaw) where the step ends, in the map frame

        Returns:
            points          (list):     [(x, y), ...] from 'start' to 'end', or None if no path was found
        '''

        points = None

        ############ ADD YOUR CODE HERE ############

        start_cell, goal_cell = self.to_cell(start[0], start[1]), self.to_cell(end[0], end[1])
        if start_cell is None or goal_cell is None:
            self.get_logger().error('Start or end pose lies outside the map')
            return None

        start_xy, end_xy = (start[0], start[1]), (end[0], end[1])
        endpoints = [(start_xy, float(self.clr[start_cell])), (end_xy, float(self.clr[goal_cell]))]

        # Rules tried in order: (drive clearance, turn clearance, every bend turn-safe).
        # Task 3B: the turn clearance is widened first, for rocks shifted from /map.
        # First the split rule with every bend on a spot with room to turn, widest berth
        # first; then the split rule allowing bends in tight spots; then the plain circle
        # rule (same clearance everywhere).
        rules = ([(drive, turn_clearance + shift, True)
                  for shift in rock_shift_margins for drive in drive_clearances[:-1]]
                 + [(drive_clearances[-1], turn_clearance, True)]
                 + [(drive_clearances[-1], turn_clearance, False)]
                 + [(robot_radius + margin, robot_radius + margin, False) for margin in fallback_margins])
        grid = (self.origin[0], self.origin[1], self.extent[0], self.extent[1], self.res)
        for rule, (drive, turn, strict) in enumerate(rules):
            # Drivable may be relaxed round the poses (the eBot is already standing there);
            # turnable never is, so the eBot only ever turns where it really has room.
            drivable = build_traversable(self.clr, drive, endpoints, self.cell_xy, endpoint_relax_radius)
            turnable = self.clr >= turn
            if not (drivable[start_cell] and drivable[goal_cell]):
                continue
            penalty = np.clip((soft_clearance - self.clr) / max(soft_clearance - drive, 1e-3), 0.0, 1.0)
            cost = 1.0 + soft_weight * penalty + tight_weight * ~turnable

            def visible(start_point, end_point):
                '''
                Description:    True when the straight line between the two points stays on
                                drivable cells.

                Args:
                    start_point     (tuple):    (x, y) start of the line
                    end_point       (tuple):    (x, y) end of the line

                Returns:
                    clear           (bool):     True if the eBot can drive straight between them
                '''
                return has_line_of_sight(start_point, end_point, drivable, grid, self.res * 0.25)

            # A tight pose is left and entered along its heading (the x axis: every allowed
            # yaw is 0 or pi), forwards or backwards, to the first spot with room to turn.
            best = None
            for start_stub in self.stubs(start_xy, start[2], visible, turnable):
                for end_stub in self.stubs(end_xy, end[2], visible, turnable):
                    middle = self.search(start_stub, end_stub, drivable, turnable, cost, visible, strict)
                    if middle is None:
                        continue
                    pts = (([start_xy] if start_stub != start_xy else []) + middle
                           + ([end_xy] if end_stub != end_xy else []))
                    if best is None or polyline_length(pts) < polyline_length(best):
                        best = pts
            if best is None:
                continue
            points = best
            if strict:
                self.get_logger().info(f'Path keeps {drive:.2f} m from the rocks, {turn:.2f} m at every bend')
            else:
                self.get_logger().warn(f'No path with every bend turn-safe; planned at {drive:.3f} m clearance')
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
            candidate_points    (list):     candidate (x, y) points; [pose] if nothing better is found
        '''
        cell = self.to_cell(*pose)
        if cell is not None and turnable[cell]:
            return [pose]
        # A point with the room asked for is preferred; failing that, one with the plain turn
        # clearance and stub_margin to spare, then one that just has room to turn.
        for wanted in (turnable, self.clr >= turn_clearance + stub_margin, self.clr >= turn_clearance):
            found = []
            for heading in (yaw, yaw + math.pi):
                for distance in np.arange(self.res, max_stub_length + 1e-9, self.res):
                    candidate = (pose[0] + distance * math.cos(heading), pose[1] + distance * math.sin(heading))
                    candidate_cell = self.to_cell(*candidate)
                    if candidate_cell is None or not visible(pose, candidate):
                        break
                    if wanted[candidate_cell]:
                        found.append(candidate)
                        break
            if found:
                return found
        return [pose]


    def search(self, start_xy, end_xy, drivable, turnable, cost, visible, strict):
        '''
        Description:    A* between two points, then pulled taut so that every bend is turn-safe.

        Args:
            start_xy        (tuple):    (x, y) start point
            end_xy          (tuple):    (x, y) end point
            drivable        (np.ndarray):   bool grid, room to drive straight
            turnable        (np.ndarray):   bool grid, room to turn on the spot
            cost            (np.ndarray):   float grid of step-cost multipliers (>= 1)
            visible         (function): (p, q) -> bool, drivable straight line
            strict          (bool):     True if every bend must have room to turn on the spot

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
        if strict:
            # Bends only where there is room to turn. The search run the other way round
            # gives a slightly different grid path, so both are tried and the shorter kept.
            taut = shortest_turn_safe(raw, turn_ok, visible, bend_stride)
            back_cells = astar(drivable, cost, end_cell, start_cell)[::-1]
            back_raw = [start_xy] + [self.to_xy(grid_cell) for grid_cell in back_cells[1:-1]] + [end_xy]
            back_ok = [True] + [bool(turnable[grid_cell]) for grid_cell in back_cells[1:-1]] + [True]
            back_taut = shortest_turn_safe(back_raw, back_ok, visible, bend_stride)
            if back_taut is not None and (taut is None or polyline_length(back_taut) < polyline_length(taut)):
                taut = back_taut
        else:
            # The greedy shortcut also accepts the tight points inside a narrow gap
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

        refined = refine_bends(taut, visible, can_turn, refine_passes, refine_tol)
        return merge_short_segments(refined, visible, min_segment_length)


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
        Description:    Moves on to the next leg of the lap, or finishes after leg 3.

        Args:

        Returns:
        '''
        self.step += 1
        self.still_since = None
        self.near_since = None
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
                if not at_start:
                    self.near_since = None
                elif self.near_since is None:
                    self.near_since = now
                timed_out = self.near_since is not None and now - self.near_since >= still_timeout_s
                if not at_start or not (self.is_still() or timed_out):
                    self.get_logger().info(
                        f'Step {self.step + 1}: waiting for the eBot to stand still at the start '
                        f'(at {self.x:.2f}, {self.y:.2f}, v={self.linear_speed:.3f}, w={self.turn_rate:.3f})',
                        throttle_duration_sec=2.0)
                    return
                start_yaw = snap_yaw(self.yaw, yaw_choices[start_pose])        # the heading it really has
                start = (start_pose[0], start_pose[1], start_yaw)
                points = self.plan_path(start, end_pose)
                if points is None:
                    self.get_logger().error(f'No path found for step {self.step + 1}; retrying',
                                            throttle_duration_sec=2.0)
                    return
                (prev_x, prev_y), (last_x, last_y) = points[-2], points[-1]      # end yaw: match the arrival
                end = (end_pose[0], end_pose[1],
                       snap_yaw(math.atan2(last_y - prev_y, last_x - prev_x), yaw_choices[end_pose]))
                self.publish_path(points, start, end)
                self.lengths.append(polyline_length(points))
                self.current_end = end
                self.current_end_pose = end_pose
                self.near_since = None
                self.state = self.STATE_DRIVING
                self.get_logger().info(f'Step {self.step + 1}: published path, {len(points)} vertices, '
                                       f'{self.lengths[-1]:.2f} m, end yaw {end[2]:.2f}')

            # ------------------------------------------------------------------
            # STATE 2: Wait for the follower to stop on the step's end pose
            # ------------------------------------------------------------------
            elif self.state == self.STATE_DRIVING:
                end_x, end_y, _ = self.current_end
                facing = min(abs(wrap_angle(self.yaw - choice)) for choice in yaw_choices[self.current_end_pose])
                near = (math.hypot(self.x - end_x, self.y - end_y) < arrive_dist and
                        facing < arrive_yaw)                               # either allowed yaw counts
                if not near:
                    self.near_since = None
                    self.still_since = None
                    return
                if self.near_since is None:
                    self.near_since = now
                if not self.is_still():
                    self.still_since = None
                elif self.still_since is None:
                    self.still_since = now
                stood_still = self.still_since is not None and now - self.still_since >= still_hold_s
                if not stood_still and now - self.near_since < still_timeout_s:
                    return
                self.get_logger().info(f'Step {self.step + 1} finished')
                if self.step == 0:
                    self.state = self.STATE_REQUEST_PACKAGE
                    self.last_spawn_try = None
                elif self.step == handover_step:
                    self.at_arm_pub.publish(Bool(data=True))
                    self.get_logger().info('Beside the arm with the package: waiting for the arm to lift it')
                    self.state = self.STATE_WAIT_HANDOVER
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

            # ------------------------------------------------------------------
            # STATE 4: Stand still beside the arm until the package is lifted
            # ------------------------------------------------------------------
            elif self.state == self.STATE_WAIT_HANDOVER:
                if self.package_lifted:
                    self.next_step()

            # STATE 5: Completed - nothing more to do

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
