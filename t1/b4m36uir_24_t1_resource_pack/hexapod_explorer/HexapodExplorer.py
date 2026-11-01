#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import numpy as np
import copy

#cpg network
import cpg.oscilator_network as osc

#import messages
from messages import *

import matplotlib.pyplot as plt

import scipy.ndimage as ndimg
from sklearn.cluster import KMeans

import skimage.measure

import collections
import heapq

NEIGHBORS_8 = [(-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
               ( 0, -1, 1.0),                        ( 0, 1, 1.0),
               ( 1, -1, math.sqrt(2)), ( 1, 0, 1.0), ( 1, 1, math.sqrt(2))]

class HexapodExplorer:

    def __init__(self):
        pass

    @staticmethod
    def world_to_grid(grid_map, pose):
        """ Convert a pose in world coordinates to (row, col) grid indices
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
            pose: Pose - pose in world coordinates
        Returns:
            (row, col): tuple of ints - row is y, col is x
        """
        col = int((pose.position.x - grid_map.origin.position.x) / grid_map.resolution)
        row = int((pose.position.y - grid_map.origin.position.y) / grid_map.resolution)

        return (row, col)

    @staticmethod
    def grid_to_world(grid_map, cell):
        """ Convert (row, col) grid indices to a pose at the center of the cell
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
            cell: tuple of ints - (row, col) grid indices
        Returns:
            pose: Pose - center of the cell in world coordinates
        """
        row, col = cell
        x = (col + 0.5) * grid_map.resolution + grid_map.origin.position.x
        y = (row + 0.5) * grid_map.resolution + grid_map.origin.position.y
        return Pose(position=Vector3(x, y, 0.0))

    @staticmethod
    def in_bounds(grid, cell):
        row, col = cell
        return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]

    @staticmethod
    def bresenham(a, b):
        """ Cells on the Bresenham line between grid cells a and b (inclusive)
        Args:
            a, b: (row, col) - line endpoints
        Returns:
            list of (row, col) - cells on the line
        """
        (r0, c0), (r1, c1) = a, b
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dc - dr
        cells = []
        r, c = r0, c0
        while True:
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr
        return cells

    def line_free(self, grid, a, b):
        """ True if the straight segement between cells a and b avoids all obstacles"""
        for cell in self.bresenham(a, b):
            if not self.in_bounds(grid, cell) or grid[cell] != 0:
                return False
        return True


    def fuse_laser_scan(self, grid_map, laser_scan, odometry):
        """ Method to fuse the laser scan data sampled by the robot with a given 
            odometry into the probabilistic occupancy grid map
        Args:
            grid_map: OccupancyGrid - gridmap to fuse te laser scan to
            laser_scan: LaserScan - laser scan perceived by the robot
            odometry: Odometry - perceived odometry of the robot
        Returns:
            grid_map_update: OccupancyGrid - gridmap updated with the laser scan data
        """
        grid_map_update = copy.deepcopy(grid_map)

        #TODO:[t1d_map] fuse the correctly aligned laser data into the probabilistic occupancy grid map

        return grid_map_update


    def find_free_edge_frontiers(self, grid_map):
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[t1e_expl] find free-adges and cluster the frontiers
        return None 


    def find_inf_frontiers(self, grid_map):
        """Method to find the frontiers based on information theory approach
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[project] find the information rich points in the environment
        return None


    def grow_obstacles(self, grid_map, robot_size):
        """ Method to grow the obstacles to take into account the robot embodiment
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            robot_size: float - size of the robot
        Returns:
            grid_map_grow: OccupancyGrid - gridmap with considered robot body embodiment
        """

        grid_map_grow = copy.deepcopy(grid_map)

        data = np.asarray(grid_map.data, dtype=float).reshape((grid_map.height, grid_map.width))
        occupied = data > 0.5
        unknown  = data == 0.5

        if occupied.any():
            distance = ndimg.distance_transform_edt(~occupied) * grid_map.resolution
        else:
            distance = np.full(data.shape, np.inf)

        blocked = occupied | unknown | (distance < robot_size)
        grid_map_grow.data = blocked.astype(float).flatten()

        return grid_map_grow


    def plan_path(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        if grid_map is None or start is None or goal is None:
            return None

        grid = np.asarray(grid_map.data).reshape((grid_map.height, grid_map.width))
        s = self.world_to_grid(grid_map, start)
        g = self.world_to_grid(grid_map, goal)

        for cell in (s, g):
            if not self.in_bounds(grid, cell) or grid[cell] > 0.5:
                return None

        # Define the heuristic function using octile distance
        def h(cell):
            dr = abs(cell[0] - g[0])
            dc = abs(cell[1] - g[1])
            return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

        # A* search algorithm
        g_cost = {s: 0.0}
        parent = {s: None}
        closed = set()
        counter = 0
        open_heap = [(h(s), counter, s)]

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == g:
                break
            closed.add(current)

            for dr, dc, step in NEIGHBORS_8:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self.in_bounds(grid, neighbor) or grid[neighbor] != 0 or neighbor in closed:
                    continue
                new_cost = g_cost[current] + step
                if new_cost < g_cost.get(neighbor, np.inf):
                    g_cost[neighbor] = new_cost
                    parent[neighbor] = current
                    counter += 1
                    heapq.heappush(open_heap, (new_cost + h(neighbor), counter, neighbor))
        else:
            return None

        cells = []
        current = g
        while current is not None:
            cells.append(current)
            current = parent[current]
        cells.reverse()

        path = Path()

        path.poses.append(start)
        
        for cell in cells[1:-1]:
            path.poses.append(self.grid_to_world(grid_map, cell))
        
        path.poses.append(goal)

        return path

    def simplify_path(self, grid_map, path):
        """ Method to simplify the found path on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        if grid_map is None or path is None or len(path.poses) == 0:
            return None

        grid = np.asarray(grid_map.data).reshape(grid_map.height, grid_map.width)
        poses = path.poses
        cells = [self.world_to_grid(grid_map, pose) for pose in poses]

        path_simplified = Path()
        path_simplified.poses.append(path.poses[0])
        
        anchor = 0
        last = len(poses) - 1
        while anchor < last:
            next = anchor + 1
            for i in range(anchor + 1, last + 1):
                if self.line_free(grid, cells[anchor], cells[i]):
                    next = i
                else:
                    break
            if next == last:
                break
            path_simplified.poses.append(poses[next])
            anchor = next
        
        path_simplified.poses.append(path.poses[-1])

        return path_simplified
 
    ###########################################################################
    #INCREMENTAL Planner
    ###########################################################################

    def plan_path_incremental(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        if not hasattr(self, 'rhs'): #first run of the function
            self.rhs = np.full((grid_map.height, grid_map.width), np.inf)
            self.g = np.full((grid_map.height, grid_map.width), np.inf)

        #TODO:[t1x-dstar] plan the incremental path between the start and the goal Pose
 
        return self.plan_path(grid_map, start, goal), self.rhs.flatten(), self.g.flatten()
