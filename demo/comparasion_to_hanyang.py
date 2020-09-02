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
AREA = 20.0  # 动画窗口大小
DISTANCE_TO_GOAL_THRESHOLD = 0.1  # 判断到达终点的距离阈值

# 计算期望的规划距离
def expectPlanningDistance(velocity):
    return 1.8 * velocity + 4.0

# 进行局部规划过程
def PlanningProcess(global_spline, init_point, goal_point, local_path_planner):
    # 验证输入的正确性
    assert(isinstance(global_spline, common.Navigation) and isinstance(init_point, common.CPoint) and isinstance(goal_point, common.CPoint))
    # 定义当前位置
    current_pose = init_point
    # 记录行驶规划
    traveling_recorder = []
    planned_path_recorder = []
    # 开始进行规划
    while True:
        # 计算局部规划期望距离
        longitude_offset = expectPlanningDistance(VELOCITY)
        local_path = local_path_planner.generateLocalPath(global_spline, current_pose, longitude_offset)
        # 记录规划的局部路径
        planned_path_recorder.append(local_path)
        # 计算局部规划路径终点与全局导航终点的距离
        distance_to_goal = np.sqrt((local_path.path_[-1].x_ - goal_point.x_) ** 2 + (local_path.path_[-1].y_ - goal_point.y_) ** 2)
        if distance_to_goal < DISTANCE_TO_GOAL_THRESHOLD:
            # 判断此时局部规划已经到达终点
            traveling_recorder.append(local_path.path_)
            break
        else:
            # 判断此时局部规划没有到达终点
            # 判断到下一次局部重规划,车辆走过的距离
            each_episode_travel_distance = VELOCITY / LOCAL_PLANNING_UPDATE_FREQUENCY
            for i, cpoint in enumerate(local_path.path_):
                if local_path.points_dis_[i] >= each_episode_travel_distance:
                    traveling_recorder.append(local_path.path_[:i])
                    current_pose = local_path.path_[i]
                    break

    return traveling_recorder, planned_path_recorder

# 显示规划过程动画
def show_animate(global_path, travel_recorder, planning_recorder, title):
    # 验证输入正确性
    assert(len(travel_recorder) == len(planning_recorder))
    # 开始显示动画
    for i in range(0, len(travel_recorder)):
        for j in range(0, len(travel_recorder[i])):
            plt.cla()
            plt.axis('equal')
            # 可视化全局导航
            plt.plot(global_path.points_x_, global_path.points_y_)
            # 可视化路径
            plt.plot(planning_recorder[i].points_x_, planning_recorder[i].points_y_, "-r")
            # 可视化当前位置
            plt.arrow(travel_recorder[i][j].x_, travel_recorder[i][j].y_, np.cos(travel_recorder[i][j].theta_), np.sin(travel_recorder[i][j].theta_), fc='b', ec='k', head_width=0.5, head_length=0.5)
            # 可视化窗口
            plt.xlim(travel_recorder[i][j].x_ - AREA, travel_recorder[i][j].x_ + AREA)
            plt.ylim(travel_recorder[i][j].y_ - AREA, travel_recorder[i][j].y_ + AREA)
            plt.title(title)
            plt.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
            plt.pause(0.0001)
    plt.close()

# 测试函数,沿全局导航从起点行驶到终点
def test():
    # 首先给出全局导航路点
    waypoints_x = [0.0, 20.0, 0.0]
    waypoints_y = [0.0, 40.0, 80.0]
    # 构建2d三次样条曲线
    global_spline = cubic_spline.CubicSpline2D(waypoints_x, waypoints_y)
    # 采样间隔
    gap = 0.1
    # 对2d三次样条曲线进行采样
    sample_s = np.arange(0.0, global_spline.s_[-1], gap)
    point_x, point_y = global_spline.calcPosition(sample_s)
    point_yaw = global_spline.calcYaw(sample_s)
    point_kappa = global_spline.calcKappa(sample_s)
    # 构建全局导航路径
    global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)

    # 给出起始位置和目标点
    init_point = global_spline.calcCPoint(global_spline.s_[0])
    goal_point = global_spline.calcCPoint(global_spline.s_[-1] - common.EPS)

    # 进行局部规划
    frenet_local_path_planner = path_planning_in_frenet.localPathPlanningFactory()
    frenet_planning_traveled_path, frenet_planned_path_recorder = PlanningProcess(global_spline, init_point, goal_point, frenet_local_path_planner)

    # 判断是否显示动画
    if ANIMATE_ON:
        # 显示动画
        show_animate(global_path, frenet_planning_traveled_path, frenet_planned_path_recorder, "Frenet Planning")

    # 利用汉阳大学2012年规划方法进行局部规划
    hanyang_local_path_planner = path_planning_hanyang.localPathPlanningFactory()
    hanyang_planning_traveled_path, hanyang_planned_path_recorder = PlanningProcess(global_spline, init_point, goal_point, hanyang_local_path_planner)

    # 判断是否显示动画
    if ANIMATE_ON:
        # 显示动画
        show_animate(global_path, hanyang_planning_traveled_path, hanyang_planned_path_recorder, "HanYang Planning")

    # 进行可视化
    # 可视化朝向随里程的变化


if __name__ == "__main__":
    test()
