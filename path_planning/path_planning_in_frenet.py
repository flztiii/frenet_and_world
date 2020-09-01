#! /usr/bin/python3
#! -*- coding:utf-8 -*-

"""

本代码用于在Frenet坐标系下跟随全局导航路径生成局部规划路径
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
    def __init__(self, debug = False, ignore_curvature_change_rate = False):
        self.debug_ = debug;
        self.ignore_curvature_change_rate_ = ignore_curvature_change_rate;
        pass
    
    # 生成局部路径
    def generateLocalPath(self, global_spline, init_point, longitude_offset, lateral_offset = 0.0, sampling_gap = 0.1, method=1):
        '''
            global_spline为全局导航路径
            init_point为局部规划的起点
        '''

        # 首先验证输入的正确性
        assert(isinstance(init_point, common.CPoint))
        # 第一步，将init_point转化到frenet坐标系下
        frenet_init_point, init_corresponding_sample = self.__transPointToFrenet(global_spline, init_point)
        print("frenet init point is ", frenet_init_point.x_, ", ", frenet_init_point.y_)
        print("frenet offset is ", frenet_init_point.x_ - global_spline.calcArcLength(init_corresponding_sample))
        # 如果是debug模式,进行起始点可视化
        if self.debug_:
            # 初始化采样间隔
            gap = 0.1
            # 对2d三次样条曲线进行采样
            sample_s = np.arange(0.0, global_spline.s_[-1], gap)
            point_x, point_y = global_spline.calcPosition(sample_s)
            point_yaw = global_spline.calcYaw(sample_s)
            point_kappa = global_spline.calcKappa(sample_s)
            # 构建全局导航路径
            global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)
            # 进行可视化
            fig = plt.figure()
            plt.axis('equal')
            plt.plot(global_path.points_x_, global_path.points_y_)
            plt.plot([global_spline.calcCPoint(init_corresponding_sample).x_], [global_spline.calcCPoint(init_corresponding_sample).y_], 'o')
            plt.plot([init_point.x_], [init_point.y_], 'o')
            plt.show()

        # 第二步, 计算目标点
        frenet_goal_point = common.CPoint(frenet_init_point.x_ + longitude_offset, lateral_offset, 0.0, 0.0)
        # 计算目标点对应的采样
        goal_corresponding_sample = global_spline.arcLengthToSample(frenet_goal_point.x_, init_corresponding_sample)
        assert(frenet_goal_point.x_ < global_spline.total_length_)
        print("frenet goal point is ", frenet_goal_point.x_, ", ", frenet_goal_point.y_)
        # 如果是debug模式,对目标点进行可视化
        if self.debug_:
            # 计算对应坐标点
            goal_world_point = global_spline.calcCPoint(goal_corresponding_sample)
            # 初始化采样间隔
            gap = 0.1
            # 对2d三次样条曲线进行采样
            sample_s = np.arange(0.0, global_spline.s_[-1], gap)
            point_x, point_y = global_spline.calcPosition(sample_s)
            point_yaw = global_spline.calcYaw(sample_s)
            point_kappa = global_spline.calcKappa(sample_s)
            # 构建全局导航路径
            global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)
            # 进行可视化
            fig = plt.figure()
            plt.axis('equal')
            plt.plot(global_path.points_x_, global_path.points_y_)
            plt.plot([init_point.x_], [init_point.y_], 'or')
            plt.plot([goal_world_point.x_], [goal_world_point.y_], 'ob')
            plt.show()
        
        # 第三步，得到frenet系下的局部规划路径
        frenet_spline = g2_spline.G2Spline(frenet_init_point, frenet_goal_point)
        sample_number = int((longitude_offset + lateral_offset) / sampling_gap)
        samples = np.linspace(0, frenet_spline.eta_, sample_number)
        frenet_path_x, frenet_path_y, frenet_path_yaw, frenet_path_curvature = [], [], [], []
        for sample in samples:
            point_x, point_y = frenet_spline.calcPosition(sample)
            point_yaw = frenet_spline.calcYaw(sample)
            point_curvature = frenet_spline.calcCurvature(sample)
            frenet_path_x.append(point_x)
            frenet_path_y.append(point_y)
            frenet_path_yaw.append(point_yaw)
            frenet_path_curvature.append(point_curvature)
        frenet_path = common.CPath(frenet_path_x, frenet_path_y, frenet_path_yaw, frenet_path_curvature)
        # 如果是debug模式,进行可视化frenet坐标系下的路径
        if self.debug_:
            # 初始化采样间隔
            gap = 0.1
            # 对2d三次样条曲线进行采样
            sample_s = np.arange(0.0, global_spline.s_[-1], gap)
            point_x, point_y = global_spline.calcPosition(sample_s)
            point_yaw = global_spline.calcYaw(sample_s)
            point_kappa = global_spline.calcKappa(sample_s)
            # 构建全局导航路径
            global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)
            # 可视化frenet坐标下的导航路径
            frenet_reference_lateral_offsets = [0.0] * len(global_path.points_dis_)
            fig = plt.figure()
            plt.axis('equal')
            plt.plot(global_path.points_dis_, frenet_reference_lateral_offsets)
            plt.plot([global_spline.calcArcLength(init_corresponding_sample)], [0.0], 'or')
            plt.plot([global_spline.calcArcLength(goal_corresponding_sample)], [0.0], 'ob')
            # 可视化生成的路径
            plt.plot(frenet_path_x, frenet_path_y)
            plt.show()
        
        # 第四步，将路径转化到world坐标系下
        # 有两种转化方法
        if method == 1:
            # 第一种是利用公式直接进行转化
            # 记录转换的时间开销
            time_start = time.time()
            world_path_x, world_path_y, world_path_yaw, world_path_curvature = [], [], [], []
            current_sample = init_corresponding_sample
            for i in range(0, len(frenet_path.path_)):
                # 计算对应的采样
                world_cpoint, corresponding_sample = self.__transPointToWorld(global_spline, frenet_path.path_[i], current_sample)
                # 更新采样点
                current_sample = corresponding_sample
                # 记录world系坐标信息
                world_path_x.append(world_cpoint.x_)
                world_path_y.append(world_cpoint.y_)
                world_path_yaw.append(world_cpoint.theta_)
                world_path_curvature.append(world_cpoint.curvature_)
                if i == 0:
                    print("init corresponding_sample", init_corresponding_sample, ", corresponding_sample ", corresponding_sample)
            local_path = common.CPath(world_path_x, world_path_y, world_path_yaw, world_path_curvature)
            time_end = time.time()
            print("frenet to world time consume: ", time_end - time_start)
            return local_path
        else:
            # 第二种，先转化坐标，再利用坐标计算曲率
            world_path_x, world_path_y, world_path_yaw, world_path_curvature = [], [], [], []
            offsets = []
            current_sample = init_corresponding_sample
            for i in range(0, len(frenet_path.path_)):
                # 计算对应的采样
                world_cpoint, corresponding_sample = self.__transPointToWorld(global_spline, frenet_path.path_[i], current_sample)
                # 更新采样点
                current_sample = corresponding_sample
                # 记录world系坐标信息
                world_path_x.append(world_cpoint.x_)
                world_path_y.append(world_cpoint.y_)
                offsets.append(frenet_path.points_x_[i] - global_spline.calcArcLength(current_sample))
                if i == 0:
                    print("init corresponding_sample", init_corresponding_sample, ", corresponding_sample ", corresponding_sample)
            if self.debug_:
                # 可视化offsets
                plt.figure()
                plt.plot(range(0, len(offsets)), offsets)
                plt.show()

            # 朝向的计算方法为atan(dy, dx)
            for i in range(0, len(world_path_x) - 1):
                # 首先计算x的变化量dx
                dx = world_path_x[i + 1] - world_path_x[i]
                # 之后计算y的变化量dy
                dy = world_path_y[i + 1] - world_path_y[i]
                # 计算朝向
                yaw = np.arctan2(dy, dx)
                world_path_yaw.append(yaw)
            world_path_yaw.append(world_path_yaw[-1])

            # 曲率的计算方法通过dyaw/ds
            for i in range(0, len(world_path_yaw) - 1):
                # 首先计算角度的变化量
                dyaw = world_path_yaw[i + 1] - world_path_yaw[i]
                # 再计算路程的变化量
                ds = np.linalg.norm(np.array([world_path_x[i + 1], world_path_y[i + 1]]) - np.array([world_path_x[i], world_path_y[i]]))
                assert ds != 0
                point_curvature = dyaw / ds
                world_path_curvature.append(point_curvature)
            world_path_curvature.append(world_path_curvature[-1])
            local_path = common.CPath(world_path_x, world_path_y, world_path_yaw, world_path_curvature)
            return local_path

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
            if (sample <= global_spline.s_[0] or sample >= global_spline.s_[-1]):
                sample = max(global_spline.s_[0] + common.EPS, min(sample, global_spline.s_[-1] - common.EPS))
        return sample

    # 将坐标从World转化到Frenet系下
    def __transPointToFrenet(self, global_spline, point):
        # 首先计算出定位对应的sample
        init_corresponding_sample = self.__calcCorrespondingSample(global_spline, point)

        # 如果有效，将坐标进行转化
        reference_point = global_spline.calcCPoint(init_corresponding_sample)
        arc_length = global_spline.calcArcLength(init_corresponding_sample)
        assert(reference_point is not None and arc_length is not None)

        # 得到world系坐标参数
        x = point.x_
        y = point.y_
        beta = point.theta_
        alpha = point.curvature_

        # 得到全局导航参考点的参数
        X, Y = global_spline.calcPosition(init_corresponding_sample)
        t = global_spline.calcYaw(init_corresponding_sample)
        k = global_spline.calcKappa(init_corresponding_sample)
        m = global_spline.calcKappaChangeRate(init_corresponding_sample)
        assert(X is not None and X == reference_point.x_)
        assert(Y is not None and Y == reference_point.y_)
        assert(t is not None and t == reference_point.theta_)
        assert(k is not None and k == reference_point.curvature_)
        assert(m is not None)

        if self.ignore_curvature_change_rate_:
            m = 0

        # 计算Frenet系坐标
        l = arc_length + (x - X) * np.cos(t) + (y - Y) * np.sin(t)
        r = (X - x) * np.sin(t) + (y - Y) * np.cos(t)
        assert(r * k != 1)
        theta = np.arctan2(np.sin(beta - t), np.cos(beta - t) / (1 - r * k))
        Q = -2 + r * k * (2 - r * k) + r * k * (-2 + r * k) * np.cos(2 * beta - 2 * t)
        curvature = np.sign(1 - r * k) * (-4 * alpha * (1 - r * k)**2 + k * (-1 + r * k) * (np.cos(3 * beta - 3 * t) - 5 * np.cos(beta - t)) + 4 * m * r * np.cos(beta - t) ** 2 * np.sin(beta - t)) / (sqrt(-2 * Q) * Q)
        frenet_point = common.CPoint(l, r, theta, curvature)

        return frenet_point, init_corresponding_sample
    
    # 将Frenet坐标转到World系下
    def __transPointToWorld(self, global_spline, point, init_sample=0.0):
        # 首先得到frenet坐标参数
        l = point.x_
        r = point.y_
        beta = point.theta_
        alpha = point.curvature_

        # 计算l对应的sample
        correponding_sample = global_spline.arcLengthToSample(l, init_sample)
        reference_point = global_spline.calcCPoint(correponding_sample)
        arc_length = global_spline.calcArcLength(correponding_sample)
        assert(arc_length is not None)

        # 得到全局导航参考点的参数
        X, Y = global_spline.calcPosition(correponding_sample)
        t = global_spline.calcYaw(correponding_sample)
        k = global_spline.calcKappa(correponding_sample)
        m = global_spline.calcKappaChangeRate(correponding_sample)
        assert(X is not None and X == reference_point.x_)
        assert(Y is not None and Y == reference_point.y_)
        assert(t is not None and t == reference_point.theta_)
        assert(k is not None and k == reference_point.curvature_)
        assert(m is not None)

        if self.ignore_curvature_change_rate_:
            m = 0

        # 计算world系坐标参数
        offset = l - arc_length
        x = X - r * np.sin(t) + offset * np.cos(t)
        y = Y + r * np.cos(t) + offset * np.sin(t)
        theta = np.arctan2(np.cos(t)*np.sin(beta)+(1 - r*k)*np.cos(beta)*np.sin(t), (1 - r*k)*np.cos(beta)*np.cos(t)-np.sin(beta)*np.sin(t))
        curvature = (alpha - r * alpha * k + k * (1 - r * k)**2 * np.cos(beta)**3 + m * r * np.cos(beta)**2 * np.sin(beta) + 2 * k * np.cos(beta) * np.sin(beta)**2) / np.power((1 - r * k)**2 * np.cos(beta)**2 + np.sin(beta)**2, 1.5)
        world_point = common.CPoint(x, y, theta, curvature)

        return world_point, correponding_sample

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
    sample_s = np.arange(0.0, global_spline.s_[-1], gap)
    point_x, point_y = global_spline.calcPosition(sample_s)
    point_yaw = global_spline.calcYaw(sample_s)
    point_kappa = global_spline.calcKappa(sample_s)
    # 构建全局导航路径
    global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)
    # 给定车辆初始位置
    init_point = common.CPoint(12.0, 15.0, 1.0, 0.0)
    # 生成局部路径
    local_path_factory = localPathPlanningFactory()
    # 第一类方法生成曲率
    local_path_1 = local_path_factory.generateLocalPath(global_spline, init_point, 40.0)
    # 第二类方法生成曲率
    local_path_2 = local_path_factory.generateLocalPath(global_spline, init_point, 40.0, method=2)

    # 计算两者的曲率最大差
    curvature_deviations = []
    for i,_ in enumerate(local_path_1.path_):
        if i < len(local_path_1.path_) - 2:
            curvature_deviations.append(abs(local_path_1.points_curvature_[i] - local_path_2.points_curvature_[i]))
    max_curvature_deviation = max(curvature_deviations)
    print("max curvature deviation is ", max_curvature_deviation, ", index is ", curvature_deviations.index(max_curvature_deviation))
    print("average curvature deviation is ", np.mean(curvature_deviations))

    # 可视化
    fig_1 = plt.figure()
    fig_1_ax = fig_1.add_subplot(1, 1, 1)
    fig_1_ax.axis('equal')
    # 可视化全局导航路径
    global_path_vis, = fig_1_ax.plot(point_x, point_y, ':')
    # 可视化局部路径(公式)
    local_path_vis_formula, = fig_1_ax.plot(local_path_1.points_x_, local_path_1.points_y_)
    # 可视化局部路径(估计)
    local_path_vis_estimate, = fig_1_ax.plot(local_path_2.points_x_, local_path_2.points_y_)
    # 添加网格
    fig_1_ax.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
    # 添加label
    fig_1_ax.set_xlabel('position[m]')
    fig_1_ax.set_ylabel('position[m]')
    # 添加标注
    fig_1_ax.legend([global_path_vis, local_path_vis_formula, local_path_vis_estimate], ['global path', 'local path formula', 'local path estimate'], loc='upper right')
    # 绘制朝向随路程的变化曲线
    fig_2 = plt.figure()
    fig_2_ax = fig_2.add_subplot(1, 1, 1)
    # 可视化local_path_1的曲率随路程的变化曲线
    local_path_1_yaw_vis, = fig_2_ax.plot(local_path_1.points_dis_, local_path_1.points_yaw_)
    # 可视化local_path_2的曲率随路程的变化曲线
    local_path_2_yaw_vis, = fig_2_ax.plot(local_path_2.points_dis_, local_path_2.points_yaw_, ':')
    # 添加标注
    fig_2_ax.legend([local_path_1_yaw_vis, local_path_2_yaw_vis], ['yaw with function', 'yaw with estimate'], loc='upper right')
    # 添加label
    fig_2_ax.set_xlabel('distance[m]')
    fig_2_ax.set_ylabel('yaw[rad]')
    # 添加标题
    fig_2_ax.set_title('yaw profile over distance')
    # 绘制曲率随路程的变化曲线
    fig_3 = plt.figure()
    fig_3_ax = fig_3.add_subplot(1, 1, 1)
    # 可视化local_path_1的曲率随路程的变化曲线
    local_path_1_cur_vis, = fig_3_ax.plot(local_path_1.points_dis_, local_path_1.points_curvature_)
    # 可视化local_path_2的曲率随路程的变化曲线
    local_path_2_cur_vis, = fig_3_ax.plot(local_path_2.points_dis_, local_path_2.points_curvature_, ':')
    # 添加标注
    fig_3_ax.legend([local_path_1_cur_vis, local_path_2_cur_vis], ['curvature with function', 'curvature with estimate'], loc='upper right')
    # 添加label
    fig_3_ax.set_xlabel('distance[m]')
    fig_3_ax.set_ylabel('curvature[rad/m]')
    # 添加标题
    fig_3_ax.set_title('curvature profile over distance')
    plt.show()

if __name__ == "__main__":
    test()