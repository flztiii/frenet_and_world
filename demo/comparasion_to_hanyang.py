#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

本代码用于比较汉阳大学规划和本规划之间的差异
author: flztiii

"""

import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from math import *
import tools.common as common
import global_path.cubic_spline as cubic_spline
import path_planning.path_planning_in_frenet as path_planning_in_frenet
import path_planning.path_planning_hanyang as path_planning_hanyang

# 全局变量
VELOCITY = 10.0  # 车辆行驶速度[m/s]
LOCAL_PLANNING_UPDATE_FREQUENCY = 10.0  # 局部规划更新频率[Hz]

# 测试函数,沿全局导航从起点行驶到终点
def test():
    pass

if __name__ == "__main__":
    test()