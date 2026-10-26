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
# Author List:      Krishang Nigam, Masum Pancholi, Harshil Jayswal, Divy Vaghasiya
# Filename:         task2B_planning.py
# Functions:        mapcb, odomcb, compute_distance_field, edt_1d, world_to_cell,
#                   cell_to_world, distance_at, search, line_is_clear, shortcut_path,
#                   resample_path, plan_path, publish_path, request_ore_package,
#                   ore_done_cb, is_still, at_pose, process_planning
# Nodes:            ebot_planner_node
#                   Publishing Topics  - [ /ebot_path ]
#                   Subscribing Topics - [ /map, /odom ]
#                   Service Clients    - [ /spawn_ore_package ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
import heapq
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_srvs.srv import Trigger


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


# ---------------- Planner tuning (change these while testing) ---------------- #

# Cells closer than this to a rock are forbidden. Tries the first value; if no path
# exists it falls back to the next ones. Larger = safer but longer path.
HARD_RADII = [0.40, 0.37, 0.34, 0.31, 0.29, 0.27]

# Path lengths (m) the evaluator reports as the benchmark for each step. For each step the
# planner takes the LARGEST hard radius (safest) whose path is no longer than this, and
# only if none is, the shortest path it found.
STEP_BENCHMARK = [6.37, 6.10, 1.90]
BENCHMARK_MARGIN = 0.02

# Soft-cost weights tried for the search; each gives a different route around the rocks
# and the shortest after tightening wins.
SOFT_WEIGHTS = [3.0, 1.0, 0.3, 0.0]

# Corner-tightening passes (pulls the path taut around rocks).
TIGHTEN_PASSES = 80

# Inside this distance of a rock the search pays an extra cost, so the path keeps away
# from rocks when that costs little extra length.
SOFT_RADIUS = 0.70
SOFT_WEIGHT = 3.0

# Shortcutting only joins two points if the straight line keeps this much more room
# than the hard radius.
SHORTCUT_EXTRA = 0.04

# Around each step's start and end, cells that are not occupied are forced free, so
# inflation from a nearby rock can never make the poses unreachable.
FORCE_FREE_RADIUS = 0.40
MIN_CLEARANCE_AT_ENDS = 0.15

# Spacing of the published path points.
PATH_SPACING = 0.15

# A cell value at or above this is a rock.
OCCUPIED_VALUE = 50


# ---------------- Sequencing tuning ---------------- #

# The eBot counts as standing still below these speeds for this long.
STILL_LINEAR = 0.02
STILL_ANGULAR = 0.02
STILL_SECONDS = 1.0

# The eBot counts as having finished a step inside these limits (task limits are
# 0.3 m and 0.15 rad).
END_POSITION_TOLERANCE = 0.25
END_YAW_TOLERANCE = 0.12

# Wait after a successful ore request for the package to land, in timer cycles.
ORE_SETTLE_CYCLES = 20

# Wait before retrying a failed ore request, in timer cycles.
ORE_RETRY_CYCLES = 20


##################### CLASS DEFINITION #######################

class ebot_planner(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to read the map, plan a path for each
                    step of the lap, publish it on /ebot_path, and request the ore package
                    at the end of step 1.
    '''

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

        # Map geometry and the precomputed costmap layers.
        self.width = 0
        self.height = 0
        self.resolution = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.dist = None                    # distance (m) from each cell to the nearest rock
        self.penalty = None                 # soft cost factor of each cell

        # Time since which the eBot has been standing still (None while moving).
        self.still_since = None

        # Sequencing state:
        #   'plan'     -> plan and publish the current step's path
        #   'driving'  -> wait for the follower to finish the step
        #   'ore'      -> request the ore package (end of step 1)
        #   'ore_wait' -> wait for the package to land
        #   'done'     -> lap finished
        self.phase = 'plan'
        self.ore_requested = False
        self.ore_success = False
        self.ore_wait_cycles = 0
        self.ore_retry_cycles = 0

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

        try:

            self.width = int(data.info.width)
            self.height = int(data.info.height)
            self.resolution = float(data.info.resolution)
            self.origin_x = float(data.info.origin.position.x)
            self.origin_y = float(data.info.origin.position.y)

            self.grid = list(data.data)

            self.compute_distance_field()

            self.get_logger().info(
                f'Map received: {self.width} x {self.height} cells, '
                f'resolution {self.resolution:.3f} m, '
                f'origin ({self.origin_x:.2f}, {self.origin_y:.2f})'
            )

        except Exception as error:

            self.get_logger().error(f'Map callback error: {error}')

            self.grid = None

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

        try:

            position = data.pose.pose.position
            orientation = data.pose.pose.orientation

            yaw = math.atan2(
                2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
            )

            self.odom = (float(position.x), float(position.y), yaw)

            linear = abs(data.twist.twist.linear.x)
            angular = abs(data.twist.twist.angular.z)

            if linear < STILL_LINEAR and angular < STILL_ANGULAR:

                if self.still_since is None:
                    self.still_since = self.get_clock().now()

            else:

                self.still_since = None

        except Exception as error:

            self.get_logger().error(f'Odometry callback error: {error}')

        ############################################


    ##################### COSTMAP #######################

    def edt_1d(self, f):
        '''
        Description:    One-dimensional squared Euclidean distance transform.

        Args:
            f (list):    Input values (0 at sources, a large number elsewhere)

        Returns:
            list:        Squared distance to the nearest source along the line
        '''

        n = len(f)
        big = 1e20

        v = [0] * n
        z = [0.0] * (n + 1)
        k = 0
        z[0] = -big
        z[1] = big

        for q in range(1, n):

            while True:

                p = v[k]
                s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0 * q - 2.0 * p)

                if s <= z[k] and k > 0:
                    k -= 1
                else:
                    break

            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = big

        out = [0.0] * n
        k = 0

        for q in range(n):

            while z[k + 1] < q:
                k += 1

            out[q] = (q - v[k]) * (q - v[k]) + f[v[k]]

        return out


    def compute_distance_field(self):
        '''
        Description:    Computes, for every cell, the distance in metres to the nearest
                        rock, and the soft cost factor derived from it.

        Args:
        Returns:
        '''

        w = self.width
        h = self.height
        big = 1e12

        # Squared distance in cells, column pass first.
        sq = [[0.0] * w for _ in range(h)]

        for col in range(w):

            column = [
                0.0 if self.grid[row * w + col] >= OCCUPIED_VALUE else big
                for row in range(h)
            ]

            result = self.edt_1d(column)

            for row in range(h):
                sq[row][col] = result[row]

        # Then the row pass.
        self.dist = [0.0] * (w * h)
        self.penalty = [0.0] * (w * h)

        for row in range(h):

            result = self.edt_1d(sq[row])

            for col in range(w):

                value = result[col]

                if value >= big:
                    d = 1e6
                else:
                    d = math.sqrt(value) * self.resolution

                index = row * w + col

                self.dist[index] = d

                self.penalty[index] = max(0.0, (SOFT_RADIUS - d) / SOFT_RADIUS)


    ##################### COORDINATES #######################

    def world_to_cell(self, x, y):
        '''
        Description:    Converts a map position to (col, row).

        Args:
            x (float):    x in metres
            y (float):    y in metres

        Returns:
            tuple:        (col, row), which may lie outside the grid
        '''

        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))

        return col, row


    def cell_to_world(self, col, row):
        '''
        Description:    Converts (col, row) to the map position of the cell centre.

        Args:
            col (int):    Column
            row (int):    Row

        Returns:
            tuple:        (x, y) in metres
        '''

        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution

        return x, y


    def distance_at(self, x, y):
        '''
        Description:    Distance to the nearest rock at a map position.
                        Outside the grid it returns 0, which counts as blocked.

        Args:
            x (float):    x in metres
            y (float):    y in metres

        Returns:
            float:        Distance in metres
        '''

        col, row = self.world_to_cell(x, y)

        if col < 0 or row < 0 or col >= self.width or row >= self.height:
            return 0.0

        return self.dist[row * self.width + col]


    ##################### SEARCH #######################

    def search(self, start, end, hard, soft_weight=None):
        '''
        Description:    A* search over the costmap with the given hard radius.

        Args:
            start (tuple):    (x, y, yaw) where the step begins
            end (tuple):      (x, y, yaw) where the step ends
            hard (float):     Cells closer than this to a rock are forbidden

        Returns:
            list:             [(x, y), ...] cell centres, or None if there is no path
        '''

        if soft_weight is None:
            soft_weight = SOFT_WEIGHT

        w = self.width
        h = self.height
        res = self.resolution

        start_col, start_row = self.world_to_cell(start[0], start[1])
        end_col, end_row = self.world_to_cell(end[0], end[1])

        for col, row in ((start_col, start_row), (end_col, end_row)):

            if col < 0 or row < 0 or col >= w or row >= h:
                return None

        # Cells forced free around the two poses (never an occupied cell).
        forced = set()
        reach = int(math.ceil(FORCE_FREE_RADIUS / res))

        for centre_col, centre_row in ((start_col, start_row), (end_col, end_row)):

            for row in range(centre_row - reach, centre_row + reach + 1):

                for col in range(centre_col - reach, centre_col + reach + 1):

                    if col < 0 or row < 0 or col >= w or row >= h:
                        continue

                    if math.hypot(col - centre_col, row - centre_row) * res > FORCE_FREE_RADIUS:
                        continue

                    index = row * w + col

                    if self.grid[index] < OCCUPIED_VALUE:
                        forced.add(index)

        dist = self.dist
        penalty = self.penalty

        def free(index):
            return dist[index] >= hard or index in forced

        start_index = start_row * w + start_col
        end_index = end_row * w + end_col

        sqrt2 = math.sqrt(2.0)

        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, sqrt2), (1, -1, sqrt2), (-1, 1, sqrt2), (-1, -1, sqrt2),
        ]

        def heuristic(col, row):
            dx = abs(col - end_col)
            dy = abs(row - end_row)
            return (max(dx, dy) + (sqrt2 - 1.0) * min(dx, dy)) * res

        g_cost = {start_index: 0.0}
        parent = {start_index: None}
        closed = set()

        queue = [(heuristic(start_col, start_row), start_index)]

        while queue:

            _, current = heapq.heappop(queue)

            if current in closed:
                continue

            closed.add(current)

            if current == end_index:
                break

            col = current % w
            row = current // w

            for d_col, d_row, length in moves:

                n_col = col + d_col
                n_row = row + d_row

                if n_col < 0 or n_row < 0 or n_col >= w or n_row >= h:
                    continue

                neighbour = n_row * w + n_col

                if neighbour in closed or not free(neighbour):
                    continue

                # No cutting corners between two rocks on a diagonal move.
                if d_col != 0 and d_row != 0:

                    if not free(row * w + n_col) or not free(n_row * w + col):
                        continue

                cost = g_cost[current] + length * res * (1.0 + soft_weight * penalty[neighbour])

                if cost < g_cost.get(neighbour, float('inf')):

                    g_cost[neighbour] = cost
                    parent[neighbour] = current

                    heapq.heappush(queue, (cost + heuristic(n_col, n_row), neighbour))

        if end_index not in parent:
            return None

        cells = []
        index = end_index

        while index is not None:

            cells.append(self.cell_to_world(index % w, index // w))
            index = parent[index]

        cells.reverse()

        return cells


    def line_is_clear(self, a, b, clear, start, end):
        '''
        Description:    Checks that a straight line keeps at least 'clear' metres from rocks.
                        Close to the step's start and end poses a smaller clearance is accepted.

        Args:
            a (tuple):        (x, y) line start
            b (tuple):        (x, y) line end
            clear (float):    Required clearance in metres
            start (tuple):    The step's start pose
            end (tuple):      The step's end pose

        Returns:
            bool:             True if the line is clear
        '''

        length = math.hypot(b[0] - a[0], b[1] - a[1])

        samples = int(length / (self.resolution * 0.5)) + 1

        for k in range(samples + 1):

            t = k / float(samples)

            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t

            d = self.distance_at(x, y)

            if d >= clear:
                continue

            near_end = (
                math.hypot(x - start[0], y - start[1]) <= FORCE_FREE_RADIUS
                or math.hypot(x - end[0], y - end[1]) <= FORCE_FREE_RADIUS
            )

            if near_end and d >= MIN_CLEARANCE_AT_ENDS:
                continue

            return False

        return True


    def shortcut_path(self, cells, clear, start, end):
        '''
        Description:    Removes unnecessary points by joining the farthest visible point.

        Args:
            cells (list):     [(x, y), ...] from the search
            clear (float):    Clearance a shortcut line must keep
            start (tuple):    The step's start pose
            end (tuple):      The step's end pose

        Returns:
            list:             The shortened [(x, y), ...]
        '''

        last = len(cells) - 1

        result = [cells[0]]

        i = 0

        while i < last:

            j = last

            while j > i + 1 and not self.line_is_clear(cells[i], cells[j], clear, start, end):
                j -= 1

            result.append(cells[j])

            i = j

        return result


    def poly_length(self, pts):
        '''
        Description:    Length of a polyline in metres.
        '''

        return sum(
            math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
            for k in range(len(pts) - 1)
        )


    def tighten_path(self, pts, clear, start, end):
        '''
        Description:    Pulls every corner of the polyline as far towards the straight line
                        between its neighbours as the clearance allows, so the path hugs
                        the rocks at exactly 'clear' instead of at the grid's coarse points.

        Args:
            pts (list):       [(x, y), ...] corner points (first and last stay fixed)
            clear (float):    Required clearance in metres
            start (tuple):    The step's start pose
            end (tuple):      The step's end pose

        Returns:
            list:             The tightened [(x, y), ...]
        '''

        pts = list(pts)

        for _ in range(TIGHTEN_PASSES):

            moved = False

            for i in range(1, len(pts) - 1):

                a = pts[i - 1]
                b = pts[i + 1]
                v = pts[i]

                abx = b[0] - a[0]
                aby = b[1] - a[1]
                denom = abx * abx + aby * aby

                if denom < 1e-12:
                    continue

                t = ((v[0] - a[0]) * abx + (v[1] - a[1]) * aby) / denom
                t = max(0.0, min(1.0, t))

                px = a[0] + abx * t
                py = a[1] + aby * t

                def candidate(f):
                    return (v[0] + (px - v[0]) * f, v[1] + (py - v[1]) * f)

                def ok(f):
                    c = candidate(f)
                    return (self.line_is_clear(a, c, clear, start, end)
                            and self.line_is_clear(c, b, clear, start, end))

                if ok(1.0):
                    best = 1.0
                else:
                    lo, hi = 0.0, 1.0

                    for _ in range(8):

                        mid = 0.5 * (lo + hi)

                        if ok(mid):
                            lo = mid
                        else:
                            hi = mid

                    best = lo

                if best < 1e-3:
                    continue

                c = candidate(best)

                before = math.hypot(v[0] - a[0], v[1] - a[1]) + math.hypot(b[0] - v[0], b[1] - v[1])
                after = math.hypot(c[0] - a[0], c[1] - a[1]) + math.hypot(b[0] - c[0], b[1] - c[1])

                if after < before - 1e-4:
                    pts[i] = c
                    moved = True

            if not moved:
                break

        return pts


    def resample_path(self, points):
        '''
        Description:    Places points at an even spacing along the polyline.

        Args:
            points (list):    [(x, y), ...] corner points

        Returns:
            list:             [(x, y), ...] evenly spaced, same first and last point
        '''

        result = [points[0]]

        for k in range(len(points) - 1):

            ax, ay = points[k]
            bx, by = points[k + 1]

            length = math.hypot(bx - ax, by - ay)

            pieces = max(1, int(math.ceil(length / PATH_SPACING)))

            for piece in range(1, pieces + 1):

                t = piece / float(pieces)

                result.append((ax + (bx - ax) * t, ay + (by - ay) * t))

        return result


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

        try:

            if self.grid is None or self.dist is None:
                return None

            target = None

            if self.step < len(STEP_BENCHMARK):
                target = STEP_BENCHMARK[self.step] - BENCHMARK_MARGIN

            best = None                     # (length, hard, corner points)

            for hard in HARD_RADII:

                found = None

                for soft_weight in SOFT_WEIGHTS:

                    cells = self.search(start, end, hard, soft_weight)

                    if cells is None:
                        break

                    cells[0] = (start[0], start[1])
                    cells[-1] = (end[0], end[1])

                    corners = cells

                    for _ in range(2):
                        corners = self.shortcut_path(corners, hard + SHORTCUT_EXTRA, start, end)
                        corners = self.tighten_path(corners, hard + SHORTCUT_EXTRA, start, end)

                    length = self.poly_length(corners)

                    if found is None or length < found[0]:
                        found = (length, hard, corners)

                if found is None:

                    self.get_logger().warning(
                        f'No path with hard radius {hard:.2f} m, trying a smaller one'
                    )

                    continue

                self.get_logger().info(
                    f'Step {self.step + 1}: hard radius {hard:.2f} m gives {found[0]:.3f} m'
                )

                if best is None or found[0] < best[0]:
                    best = found

                # Largest radius that already meets the benchmark is the safest good path.
                if target is not None and found[0] <= target:
                    best = found
                    break

            if best is not None:

                length, hard, corners = best

                points = self.resample_path(corners)

                points[0] = (start[0], start[1])
                points[-1] = (end[0], end[1])

                self.get_logger().info(
                    f'Planned path: {len(points)} points, {self.poly_length(points):.3f} m, '
                    f'hard radius {hard:.2f} m'
                )

        except Exception as error:

            self.get_logger().error(f'Planning error: {error}')

            points = None

        ############################################

        return points


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

        last = len(points) - 1

        for k, (x, y) in enumerate(points):

            if k == 0:
                yaw = start[2]
            elif k == last:
                yaw = end[2]
            else:
                yaw = math.atan2(points[k + 1][1] - y, points[k + 1][0] - x)

            pose = PoseStamped()
            pose.header.frame_id = map_frame
            pose.header.stamp = path.header.stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)

            path.poses.append(pose)

        self.path_pub.publish(path)

        self.get_logger().info(
            f'Published path for step {self.step + 1}: {len(path.poses)} poses'
        )

        ############################################


    def request_ore_package(self):
        '''
        Description:    Calls /spawn_ore_package once the eBot stands still on the ore drop pose.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        if not self.spawn_client.service_is_ready():

            self.get_logger().warning('Service /spawn_ore_package is not ready yet')

            return

        self.ore_requested = True

        future = self.spawn_client.call_async(Trigger.Request())

        future.add_done_callback(self.ore_done_cb)

        self.get_logger().info('Requested the ore package')

        ############################################


    def ore_done_cb(self, future):
        '''
        Description:    Receives the reply of /spawn_ore_package.

        Args:
            future (Future):    The finished service call

        Returns:
        '''

        try:

            result = future.result()

            self.get_logger().info(
                f'Ore package reply: success={result.success}, message="{result.message}"'
            )

            if result.success:

                self.ore_success = True

            else:

                # Try again after a pause.
                self.ore_requested = False
                self.ore_retry_cycles = 0

        except Exception as error:

            self.get_logger().error(f'Ore package call failed: {error}')

            self.ore_requested = False
            self.ore_retry_cycles = 0


    ##################### SEQUENCING HELPERS #######################

    def is_still(self, seconds):
        '''
        Description:    Tells whether the eBot has been standing still for 'seconds'.

        Args:
            seconds (float):    Required time in seconds

        Returns:
            bool:               True if it has stood still that long
        '''

        if self.still_since is None:
            return False

        elapsed = (self.get_clock().now() - self.still_since).nanoseconds * 1e-9

        return elapsed >= seconds


    def at_pose(self, pose, either_yaw=True):
        '''
        Description:    Tells whether the eBot is on 'pose'.

        Args:
            pose (tuple):         (x, y, yaw)
            either_yaw (bool):    True if yaw 0 or pi is valid (ore drop and arm poses),
                                  False if only the pose's own yaw is valid (home)

        Returns:
            bool:                 True if within the end tolerances
        '''

        x, y, yaw = self.odom

        if math.hypot(x - pose[0], y - pose[1]) > END_POSITION_TOLERANCE:
            return False

        error_zero = abs(math.atan2(math.sin(yaw - pose[2]), math.cos(yaw - pose[2])))

        if not either_yaw:
            return error_zero <= END_YAW_TOLERANCE

        error_pi = abs(math.atan2(math.sin(yaw - pose[2] - math.pi),
                                  math.cos(yaw - pose[2] - math.pi)))

        return min(error_zero, error_pi) <= END_YAW_TOLERANCE


    def process_planning(self):
        '''
        Description:    Timer function used to plan, publish and request, step by step.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            if self.grid is None or self.dist is None or self.odom is None:
                return

            if self.phase == 'done':
                return

            start, end = steps[self.step]

            # ---- Plan and publish while the eBot stands still at the step's start ----
            if self.phase == 'plan':

                if not self.is_still(0.5):
                    return

                points = self.plan_path(start, end)

                if points is None:

                    self.get_logger().error(
                        f'No path found for step {self.step + 1}',
                        throttle_duration_sec=5.0
                    )

                    return

                self.publish_path(points, start, end)

                self.phase = 'driving'

                return

            # ---- Wait for the follower to finish the step ----
            if self.phase == 'driving':

                # Home accepts only its own yaw; the ore drop and arm poses accept 0 or pi.
                either_yaw = (end != home_pose)

                if not (self.at_pose(end, either_yaw) and self.is_still(STILL_SECONDS)):
                    return

                self.get_logger().info(f'Step {self.step + 1} finished')

                if self.step == 0:

                    self.phase = 'ore'
                    self.ore_requested = False
                    self.ore_success = False
                    self.ore_retry_cycles = ORE_RETRY_CYCLES

                    return

                self.step += 1

                if self.step >= len(steps):

                    self.phase = 'done'

                    self.get_logger().info('Lap complete')

                    return

                self.phase = 'plan'

                return

            # ---- Request the ore package at the end of step 1 ----
            if self.phase == 'ore':

                if self.ore_success:

                    self.phase = 'ore_wait'
                    self.ore_wait_cycles = 0

                    return

                if not self.ore_requested:

                    self.ore_retry_cycles += 1

                    if self.ore_retry_cycles >= ORE_RETRY_CYCLES:

                        if self.at_pose(end) and self.is_still(STILL_SECONDS):
                            self.request_ore_package()

                        self.ore_retry_cycles = 0

                return

            # ---- Let the package land, then start step 2 ----
            if self.phase == 'ore_wait':

                self.ore_wait_cycles += 1

                if self.ore_wait_cycles >= ORE_SETTLE_CYCLES:

                    self.step = 1
                    self.phase = 'plan'

                return

        except Exception as error:

            self.get_logger().error(f'Planning cycle error: {error}')

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_planner class to perform its task
    '''

    rclpy.init(args=sys.argv)                                           # initialisation

    ebot_planner_class = ebot_planner()                                 # creating a new object for class 'ebot_planner'

    ebot_planner_class.get_logger().info('Node created: eBot planner process')

    try:

        rclpy.spin(ebot_planner_class)                                  # spining on the object to make it alive in ROS 2 DDS

    except KeyboardInterrupt:

        pass

    finally:

        ebot_planner_class.destroy_node()                               # destroy node after spin ends

        if rclpy.ok():
            rclpy.shutdown()                                            # shutdown process


if __name__ == '__main__':

    main()
