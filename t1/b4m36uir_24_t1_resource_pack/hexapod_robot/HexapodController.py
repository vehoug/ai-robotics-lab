#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#import messages
from messages import *

DELTA_DISTANCE = 0.12
C_TURNING_SPEED = 5
C_AVOID_SPEED = 10
ORIENTATION_THRESHOLD = np.pi/16

class HexapodController:
    def __init__(self):
        pass


    def goto(self, goal, odometry, collision):
        """Method to steer the robot towards the goal position given its current 
           odometry and collision status
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
        Returns:
            cmd: Twist steering command
        """
        
        # Zero velocity steering command
        cmd_msg = Twist()
 
        if goal is None or odometry is None or collision is None:
            return cmd_msg
 
        if collision:
            return None
 
        pos_robot = np.array([odometry.pose.position.x, odometry.pose.position.y])
        pos_goal = np.array([goal.position.x, goal.position.y])
        distance_to_goal = np.linalg.norm(pos_goal - pos_robot)
 
        if distance_to_goal < DELTA_DISTANCE:
            return None
 
        phi_robot = odometry.pose.orientation.to_Euler()[0]
        phi_goal = np.arctan2(pos_goal[1] - pos_robot[1], pos_goal[0] - pos_robot[0])
        delta_phi = phi_goal - phi_robot
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi  # Restrict to [-pi, pi)
 
        if abs(delta_phi) > ORIENTATION_THRESHOLD:
            cmd_msg.angular.z = 1.0 if delta_phi > 0 else -1.0
        else:
            cmd_msg.linear.x = 1.0
 
        return cmd_msg


    def goto_reactive(self, goal, odometry, collision, laser_scan):
        """Method to steer the robot towards the goal position while avoiding 
           contact with the obstacles given its current odometry, collision 
           status and laser scan data
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
            laser_scan: LaserScan data perceived by the robot
        Returns:
            cmd: Twist steering command
        """
        # Zero velocity steering command
        cmd_msg = Twist()

        if goal is None or odometry is None or collision is None or laser_scan is None:
            return cmd_msg

        if collision:
            return None

        pos_robot = np.array([odometry.pose.position.x, odometry.pose.position.y])
        pos_goal = np.array([goal.position.x, goal.position.y])
        distance_to_goal = np.linalg.norm(pos_goal - pos_robot)

        if distance_to_goal < DELTA_DISTANCE:
            return None

        phi_robot = odometry.pose.orientation.to_Euler()[0]
        phi_goal = np.arctan2(pos_goal[1] - pos_robot[1], pos_goal[0] - pos_robot[0])
        delta_phi = phi_goal - phi_robot
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi  # Restrict to [-pi, pi)

        distances = np.array(laser_scan.distances, dtype=float)
        valid = (distances >= laser_scan.range_min) & (distances <= laser_scan.range_max)
        distances[~valid] = np.inf
        half = len(distances) // 2

        scan_right = np.min(distances[:half], initial=np.inf)
        scan_left = np.min(distances[half:], initial=np.inf)

        repulsive_force = 1.0 / scan_right - 1.0 / scan_left

        cmd_msg.linear.x = distance_to_goal
        cmd_msg.angular.z = C_TURNING_SPEED * delta_phi + C_AVOID_SPEED * repulsive_force
        
        return cmd_msg
