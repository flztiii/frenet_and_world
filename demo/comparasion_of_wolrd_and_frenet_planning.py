#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

本代码用于比较在World系下进行规划和在Frenet坐标系下进行规划之间的差异
author: flztiii

"""

import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from math import *
import tools.common as common
import path_planning.g2_spline as g2_spline
import global_path.cubic_spline as cubic_spline
import path_planning.path_planning_in_frenet as path_planning_in_frenet
import path_planning.path_planning_in_world as path_planning_in_world

# 计算给出坐标在全局导航的对应点
def calcCorrespondingSample(global_spline, point, init_sample = 0.0):
    # 定义函数
    def func(sample):
        return 2 * (global_spline.spline_x_.calc(sample) - point.x_) * global_spline.spline_x_.calcd(sample) + 2 * (global_spline.spline_y_.calc(sample) - point.y_) * global_spline.spline_y_.calcd(sample)
    # 定义导数
    def derivate(sample):
        return 2 * (global_spline.spline_x_.calc(sample) - point.x_) * global_spline.spline_x_.calcdd(sample) + 2.0 * np.power(global_spline.spline_x_.calcd(sample), 2) + 2 * (global_spline.spline_y_.calc(sample) - point.y_) * global_spline.spline_y_.calcdd(sample) + 2.0 * np.power(global_spline.spline_y_.calcd(sample), 2)
    
    sample = init_sample
    # 进行牛顿迭代
    while np.abs(func(sample)) > 1e-3 and derivate(sample) != 0:
        sample += - func(sample) / derivate(sample)
        if (sample <= global_spline.s_[0] or sample >= global_spline.s_[-1]):
            sample = max(global_spline.s_[0] + common.EPS, min(sample, global_spline.s_[-1] - common.EPS))
    return sample

# 测试在frenet系和world系进行规划的差异
def test():
    # 首先给出全局导航路点
    waypoints_x = [0.0, 20.0, 0.0]
    waypoints_y = [0.0, 20.0, 80.0]
    # 初始化采样间隔
    gap = 0.1
    # 构建2d三次样条曲线
    global_spline = cubic_spline.CubicSpline2D(waypoints_x, waypoints_y)
    # 对2d三次样条曲线进行采样
    sample_s = np.arange(0.0, global_spline.s_[-1], gap)
    point_x, point_y = global_spline.calcPosition(sample_s)
    point_yaw = global_spline.calcYaw(sample_s)
    point_kappa = global_spline.calcKappa(sample_s)
    # 构建全局导航路径
    global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)

    # 构建局部规划器(Frenet坐标系)
    local_path_planner_in_frenet = path_planning_in_frenet.localPathPlanningFactory()
    # 构建局部规划器(World坐标系)
    local_path_planner_in_world = path_planning_in_world.localPathPlanningFactory()

    # 给出起始点
    init_point = common.CPoint(20.0, 20.0, 0.0, 0.0)

    # 记录不同的规划长度
    planning_distances = []
    # 记录规划的路径
    planned_paths_in_frenet = []
    planned_paths_in_world = []
    # 记录全局导航与局部规划之间的差异
    planning_in_frenet_avg_deviations = []
    planning_in_world_avg_deviations = []
    # 遍历规划期望距离
    for expected_planning_distance in np.linspace(5.0, 50.0, 45):
        # 记录不同的规划长度
        planning_distances.append(expected_planning_distance)
        # 进行规划
        planned_path_in_frenet = local_path_planner_in_frenet.generateLocalPath(global_spline, init_point, expected_planning_distance)
        planned_path_in_world = local_path_planner_in_world.generateLocalPath(global_spline, init_point, expected_planning_distance)
        # 记录规划的路径
        planned_paths_in_frenet.append(planned_path_in_frenet)
        planned_paths_in_world.append(planned_path_in_world)

        # 计算frenet规划与导航的偏差
        deviation = 0.0
        for i in range(0, len(planned_path_in_frenet.path_)):
            point = planned_path_in_frenet.path_[i]
            # 计算对应的全局导航点
            corresponding_sample = calcCorrespondingSample(global_spline, point)
            corresponding_point = global_spline.calcCPoint(corresponding_sample)
            # 计算与导航点的偏差
            deviation += (point - corresponding_point).value()
        deviation = deviation / expected_planning_distance
        planning_in_frenet_avg_deviations.append(deviation)

        # 计算world规划与导航的偏差
        deviation = 0.0
        for i in range(0, len(planned_path_in_world.path_)):
            point = planned_path_in_world.path_[i]
            # 计算对应的全局导航点
            corresponding_sample = calcCorrespondingSample(global_spline, point)
            corresponding_point = global_spline.calcCPoint(corresponding_sample)
            # 计算与导航点的偏差
            deviation += (point - corresponding_point).value()
        deviation = deviation / expected_planning_distance
        planning_in_world_avg_deviations.append(deviation)
        

    # 进行路径的可视化对比
    for i in range(0, len(planning_distances)):
        plt.cla()
        plt.axis('equal')
        global_path_vis, = plt.plot(point_x, point_y, ":")
        planned_path_in_frenet_vis, = plt.plot(planned_paths_in_frenet[i].points_x_, planned_paths_in_frenet[i].points_y_, "r")
        planned_path_in_world_vis, = plt.plot(planned_paths_in_world[i].points_x_, planned_paths_in_world[i].points_y_, "b")
        # 添加网格
        plt.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
        # 添加label
        plt.xlabel('position[m]')
        plt.ylabel('position[m]')
        # 添加标注
        plt.legend([global_path_vis, planned_path_in_frenet_vis, planned_path_in_world_vis], ['global path', 'local path planned in Frenet', 'local path planned in World'], loc='upper right')
        plt.pause(0.01)

    # 进行误差的对比
    plt.figure()
    planning_in_frenet_avg_deviations_vis, = plt.plot(planning_distances, planning_in_frenet_avg_deviations, "r")
    planning_in_world_avg_deviations_vis, = plt.plot(planning_distances, planning_in_world_avg_deviations, "b")
    # 添加网格
    plt.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
    # 添加label
    plt.xlabel('planning distance[m]')
    plt.ylabel('consistency deviation[m]')
    # 添加标注
    plt.legend([planning_in_frenet_avg_deviations_vis, planning_in_world_avg_deviations_vis], ['path planned in Frenet consistency deviation', 'path planned in World consistency deviation'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    test()