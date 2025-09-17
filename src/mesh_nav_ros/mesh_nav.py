#!/usr/bin/env python3
import rospy
import numpy as np


import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

# import other created files
from mesh_nav_ros.robot_mesh_state import RobotMeshState
from mesh_nav_ros.collision_detections import *
from mesh_nav_ros.graph_functions import GraphManager
from mesh_nav_ros.elastic_bands_threads import ElasticBandPlanner as ElasticBandPlannerPy
from mesh_nav_ros.robot_kinematics import RobotKinematics
from mesh_nav_ros.aux_functions import interpolate_path, sample_path

# NEW: C++ elastic band planner binding (pybind11)
import os, sys

ws = os.path.abspath(os.path.dirname(__file__))
print(ws)
for _ in range(10):  # walk up a few levels
    cand = os.path.join(ws, 'devel', 'lib')
    if os.path.isdir(cand):
        if cand not in sys.path:
            sys.path.append(cand)
        break
    parent = os.path.dirname(ws)
    if parent == ws:
        break
    ws = parent
try:
    from elastic_band_planner_cpp import ElasticBandPlanner as ElasticBandPlannerCPP
    _HAS_CPP_EBAND = True
except Exception as e:
    rospy.logwarn("Could not import C++ ElasticBandPlanner binding: %s", e)
    _HAS_CPP_EBAND = False

class MeshNav(object):
    def __init__(self):
        self.path = 'data/'
        
        