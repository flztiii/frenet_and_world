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
ANIMATE_ON = False  # 是否播放动画

# 计算期望的规划距离
def expectPlanningDistance(velocity):
    return 3.6 * velocity + 4.0

# 在frenet系进行局部规划
def frenetPlanningProcess(global_spline, init_point):
    # 验证输入的正确性
    assert(isinstance(global_spline, common.Navigation) and isinstance(init_point, common.CPoint))
    # 定义当前位置
    current_pose = init_point
    # 记录行驶规划
    traveling_recorder = []
    # 开始进行规划
    while True:
        # 计算局部规划期望距离
        longitude_offset = expectPlanningDistance(VELOCITY)
        

# 测试函数,沿全局导航从起点行驶到终点
def test():
    # 首先给出全局导航路点
    waypoints_x = [0.0, 20.0, 0.0, -20.0, 0.0, 20.0]
    waypoints_y = [0.0, 40.0, 80.0, 120.0, 160.0, 200.0]
    # 构建2d三次样条曲线
    global_spline = cubic_spline.CubicSpline2D(waypoints_x, waypoints_y)

    # 给出起始位置
    init_point = global_spline.calcCPoint(0.0)

    # 进行局部规划
    frenetPlanningProcess(global_spline, init_point)

if __name__ == "__main__":
    test()