#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This launch file starts the four Task 3A nodes of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task3A_nodes.launch.py
# Functions:        generate_launch_description
# Nodes:            task3A_perception.py, task3A_manipulation.py, task3A_planning.py,
#                   task3A_navigation.py (all from the algorithms package)
#
# Usage, once the evaluator prints Ready (the world is launched on its own, not from here):
#                   ros2 launch algorithms task3A_nodes.launch.py


################### IMPORT MODULES #######################

from launch import LaunchDescription
from launch_ros.actions import Node


##################### TASK CONSTANTS #######################

# The four Task 3A scripts, as installed by algorithms/setup.py
TASK3A_SCRIPTS = [
    'task3A_perception.py',
    'task3A_manipulation.py',
    'task3A_planning.py',
    'task3A_navigation.py',
]


##################### FUNCTION DEFINITION #######################

def generate_launch_description():
    '''
    Description:    Builds the launch description that starts the four Task 3A nodes, each
                    printing its log to this terminal.

    Args:

    Returns:
        description     (LaunchDescription):    the four nodes to start
    '''
    nodes = [Node(package='algorithms', executable=script, output='screen', emulate_tty=True)
             for script in TASK3A_SCRIPTS]
    return LaunchDescription(nodes)
