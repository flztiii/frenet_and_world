#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

本代码用于在World坐标系下跟随全局导航路径生成局部规划路径
author: flztiii

"""

import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from math import *
import time
import tools.common as common
import path_planning.g2_spline as g2_spline
import global_path.cubic_spline as cubic_spline

# 给定导航路径，定位信息，生成局部路径
class localPathPlanningFactory:
    def __init__(self):
        pass
    
    # 生成局部路径
    def generateLocalPath(self, global_spline, init_point, longitude_offset, lateral_offset = 0.0, sampling_gap = 0.1):
        '''
            global_spline为全局导航路径
            init_point为局部规划的起点
        '''
        # 首先验证输入的正确性
        assert(isinstance(global_spline, common.Navigation) and isinstance(init_point, common.CPoint))

        # 第一步,找到当前定位对应的全局导航参考点
        init_corresponding_sample = self.__calcCorrespondingSample(global_spline, init_point)
        init_longitude_offset = global_spline.calcArcLength(init_corresponding_sample)

        # 第二步,找到目标点在全局导航的参考点
        goal_reference_longitude_offset = min(init_longitude_offset + longitude_offset, global_spline.getTotalLength())
        goal_reference_corresponding_sample = global_spline.arcLengthToSample(goal_reference_longitude_offset, init_corresponding_sample)
        assert(goal_reference_corresponding_sample is not None)
        goal_reference_point = global_spline.calcCPoint(goal_reference_corresponding_sample)
        
        # 第三步,得到目标点
        assert(goal_reference_point.curvature_**(-1) != lateral_offset)
        goal_point = common.CPoint(goal_reference_point.x_ + lateral_offset * np.cos(goal_reference_point.theta_ + np.pi * 0.5), goal_reference_point.y_ + lateral_offset * np.sin(goal_reference_point.theta_ + np.pi * 0.5), goal_reference_point.theta_, (goal_reference_point.curvature_**(-1) - lateral_offset)**(-1))

        # 第四步,进行规划
        local_spline = g2_spline.G2Spline(init_point, goal_point)
        sample_number = int((longitude_offset + lateral_offset) / sampling_gap)
        samples = np.linspace(0, local_spline.eta_, sample_number)
        world_path_x, world_path_y, world_path_yaw, world_path_curvature = [], [], [], []
        for sample in samples:
            point_x, point_y = local_spline.calcPosition(sample)
            point_yaw = local_spline.calcYaw(sample)
            point_curvature = local_spline.calcCurvature(sample)
            world_path_x.append(point_x)
            world_path_y.append(point_y)
            world_path_yaw.append(point_yaw)
            world_path_curvature.append(point_curvature)
        world_path = common.CPath(world_path_x, world_path_y, world_path_yaw, world_path_curvature)
        
        return world_path

    # 计算给出坐标在全局导航的对应点
    def __calcCorrespondingSample(self, global_spline, point):
        # 定义函数
        def func(sample):
            return 2 * (global_spline.spline_x_.calc(sample) - point.x_) * global_spline.spline_x_.calcd(sample) + 2 * (global_spline.spline_y_.calc(sample) - point.y_) * global_spline.spline_y_.calcd(sample)
        # 定义导数
        def derivate(sample):
            return 2 * (global_spline.spline_x_.calc(sample) - point.x_) * global_spline.spline_x_.calcdd(sample) + 2.0 * np.power(global_spline.spline_x_.calcd(sample), 2) + 2 * (global_spline.spline_y_.calc(sample) - point.y_) * global_spline.spline_y_.calcdd(sample) + 2.0 * np.power(global_spline.spline_y_.calcd(sample), 2)
        
        sample = 0.0
        # 进行牛顿迭代
        while np.abs(func(sample)) > 1e-3 and derivate(sample) != 0:
            sample += - func(sample) / derivate(sample)
            if (sample <= global_spline.minSample() or sample >= global_spline.maxSample()):
                sample = max(global_spline.minSample() + common.EPS, min(sample, global_spline.maxSample() - common.EPS))
        return sample

# 测试函数
def test():
    # 首先建立全局导航路径
    # 初始化散点
    x = [0.0, 20.0, 0.0]
    y = [0.0, 20.0, 80.0]
    # 初始化采样间隔
    gap = 0.1
    # 构建2d三次样条曲线
    global_spline = cubic_spline.CubicSpline2D(x, y)
    # 对2d三次样条曲线进行采样
    sample_s = np.arange(0.0, global_spline.maxSample(), gap)
    point_x, point_y = global_spline.calcPosition(sample_s)
    point_yaw = global_spline.calcYaw(sample_s)
    point_kappa = global_spline.calcKappa(sample_s)
    # 构建全局导航路径
    global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)
    # 给定车辆初始位置
    init_point = common.CPoint(12.0, 15.0, 1.0, 0.0)
    # 生成局部路径
    local_path_factory = localPathPlanningFactory()
    local_path = local_path_factory.generateLocalPath(global_spline, init_point, 40.0)

    # 可视化
    fig_1 = plt.figure()
    fig_1_ax = fig_1.add_subplot(1, 1, 1)
    fig_1_ax.axis('equal')
    # 可视化全局导航路径
    global_path_vis, = fig_1_ax.plot(point_x, point_y, ':')
    # 可视化局部路径(公式)
    local_path_vis, = fig_1_ax.plot(local_path.points_x_, local_path.points_y_)
    # 添加网格
    fig_1_ax.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
    # 添加label
    fig_1_ax.set_xlabel('position[m]')
    fig_1_ax.set_ylabel('position[m]')
    # 添加标注
    fig_1_ax.legend([global_path_vis, local_path_vis], ['global path', 'local path'], loc='upper right')
    # 绘制朝向随路程的变化曲线
    fig_2 = plt.figure()
    fig_2_ax = fig_2.add_subplot(1, 1, 1)
    # 可视化local_path_1的曲率随路程的变化曲线
    local_path_yaw_vis, = fig_2_ax.plot(local_path.points_dis_, local_path.points_yaw_)
    # 添加标注
    fig_2_ax.legend([local_path_yaw_vis], ['yaw'], loc='upper right')
    # 添加label
    fig_2_ax.set_xlabel('distance[m]')
    fig_2_ax.set_ylabel('yaw[rad]')
    # 添加标题
    fig_2_ax.set_title('yaw profile over distance')
    # 绘制曲率随路程的变化曲线
    fig_3 = plt.figure()
    fig_3_ax = fig_3.add_subplot(1, 1, 1)
    # 可视化local_path_1的曲率随路程的变化曲线
    local_path_cur_vis, = fig_3_ax.plot(local_path.points_dis_, local_path.points_curvature_)
    # 添加标注
    fig_3_ax.legend([local_path_cur_vis], ['curvature'], loc='upper right')
    # 添加label
    fig_3_ax.set_xlabel('distance[m]')
    fig_3_ax.set_ylabel('curvature[rad/m]')
    # 添加标题
    fig_3_ax.set_title('curvature profile over distance')
    plt.show()

if __name__ == "__main__":
    test()
