#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

本代码用于比较带曲率变化率的转化和不带曲率变化率的转化之间的差异
author: flztiii

"""

import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from math import *
import tools.common as common
import global_path.cubic_spline as cubic_spline
import global_path.quartic_spline as quartic_spline
import path_planning.path_planning_in_frenet as path_planning_in_frenet

# 利用插值进行全局导航曲线生成
def test():
    # 首先给出全局导航路点
    waypoints_x = [0.0, 20.0, 0.0]
    waypoints_y = [0.0, 20.0, 80.0]
    # 初始化采样间隔
    gap = 0.1
    # 构建2d三次样条曲线
    global_spline = quartic_spline.QuarticSpline2D(waypoints_x, waypoints_y)
    # 对2d三次样条曲线进行采样
    sample_s = np.arange(0.0, global_spline.maxSample(), gap)
    point_x, point_y = global_spline.calcPosition(sample_s)
    point_yaw = global_spline.calcYaw(sample_s)
    point_kappa = global_spline.calcKappa(sample_s)
    # 构建全局导航路径
    global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)

    # 构建局部规划器(带曲率变化率)
    local_path_planning_with_curvature_change_rate_factory = path_planning_in_frenet.localPathPlanningFactory(ignore_curvature_change_rate=False)
    # 构建局部规划器(不带曲率变化率)
    local_path_planning_without_curvature_change_rate_factory = path_planning_in_frenet.localPathPlanningFactory(ignore_curvature_change_rate=True)

    # 给出当前位置
    init_point = common.CPoint(12.0, 15.0, 1.0, 0.0)
    # 给出规划距离
    expected_planning_distance = 40.0
    # 以带曲率变化率的规划器进行规划
    local_path_planned_with_curvature_change_rate = local_path_planning_with_curvature_change_rate_factory.generateLocalPath(global_spline, init_point, expected_planning_distance)
    # 以不带曲率变化率的规划器进行规划
    local_path_planned_without_curvature_change_rate = local_path_planning_without_curvature_change_rate_factory.generateLocalPath(global_spline, init_point, expected_planning_distance)
    # 得到ground truth
    ground_truth_local_path_1 = local_path_planning_with_curvature_change_rate_factory.generateLocalPath(global_spline, init_point, expected_planning_distance, method=2)
    ground_truth_local_path_2 = local_path_planning_without_curvature_change_rate_factory.generateLocalPath(global_spline, init_point, expected_planning_distance, method=2)

    # 验证ground truth的正确性
    assert(len(ground_truth_local_path_1.path_) == len(ground_truth_local_path_2.path_))

    # 输出数据对比
    curvature_deviation_with_curvature_change_rate, curvature_deviation_without_curvature_change_rate = [], []

    for i in range(0, len(ground_truth_local_path_1.path_) - 2):
        curvature_deviation_with_curvature_change_rate.append(np.abs(local_path_planned_with_curvature_change_rate.points_curvature_[i] - ground_truth_local_path_1.points_curvature_[i]))
        curvature_deviation_without_curvature_change_rate.append(np.abs(local_path_planned_without_curvature_change_rate.points_curvature_[i] - ground_truth_local_path_2.points_curvature_[i]))
    print("with curvature change rate planning max curvature deviation is ", max(curvature_deviation_with_curvature_change_rate), ", without curvature change rate planning max curvature deviation is ", max(curvature_deviation_without_curvature_change_rate))
    print("with curvature change rate planning mean curvature deviation is ", np.mean(curvature_deviation_with_curvature_change_rate), ", without curvature change rate planning mean curvature deviation is ", np.mean(curvature_deviation_without_curvature_change_rate))

    # 设置可视化参数
    del matplotlib.font_manager.weight_dict['roman']
    matplotlib.font_manager._rebuild()
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = [3.45, 1.6]
    plt.rcParams['figure.subplot.top'] = 0.98
    plt.rcParams['figure.subplot.bottom'] = 0.28
    plt.rcParams['figure.subplot.left'] = 0.16
    plt.rcParams['figure.subplot.right'] = 0.96
    plt.rcParams['font.family'] = "Times New Roman"
    plt.rcParams['font.size'] = 8

    # 进行可视化
    fig_1 = plt.figure()
    fig_1_ax = fig_1.add_subplot(1, 1, 1)
    # 可视化local_path_1的曲率随路程的变化曲线
    local_path_calc_with_cur_change_rate_cur_vis, = fig_1_ax.plot(local_path_planned_with_curvature_change_rate.points_dis_[:-2], local_path_planned_with_curvature_change_rate.points_curvature_[:-2])
    # 可视化local_path_2的曲率随路程的变化曲线
    local_path_ground_truth_1_cur_vis, = fig_1_ax.plot(ground_truth_local_path_1.points_dis_[:-2], ground_truth_local_path_1.points_curvature_[:-2], ':')
    # 添加标注
    fig_1_ax.legend([local_path_calc_with_cur_change_rate_cur_vis, local_path_ground_truth_1_cur_vis], ['Calculation', 'Ground truth'], loc='lower right')
    # 添加label
    fig_1_ax.set_xlabel('Distance[m]')
    fig_1_ax.set_ylabel('Curvature[rad/m]')
    # 添加标题
    # fig_1_ax.set_title('Curvature profile over distance')

    fig_2 = plt.figure()
    fig_2_ax = fig_2.add_subplot(1, 1, 1)
    # 可视化local_path_1的曲率随路程的变化曲线
    local_path_calc_without_cur_change_rate_cur_vis, = fig_2_ax.plot(local_path_planned_without_curvature_change_rate.points_dis_[:-2], local_path_planned_without_curvature_change_rate.points_curvature_[:-2])
    # 可视化local_path_2的曲率随路程的变化曲线
    local_path_ground_truth_2_cur_vis, = fig_2_ax.plot(ground_truth_local_path_2.points_dis_[:-2], ground_truth_local_path_2.points_curvature_[:-2], ':')
    # 添加标注
    fig_2_ax.legend([local_path_calc_without_cur_change_rate_cur_vis, local_path_ground_truth_2_cur_vis], ['Calculation', 'Ground truth'], loc='lower right')
    # 添加label
    fig_2_ax.set_xlabel('Distance[m]')
    fig_2_ax.set_ylabel('Curvature[rad/m]')
    # 添加标题
    # fig_2_ax.set_title('Curvature profile over distance')

    plt.show()


if __name__ == "__main__":
    test()