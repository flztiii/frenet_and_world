#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

本代码用于复现汉阳大学2012年无人驾驶局部路径规划方法
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
import global_path.cubic_spline as cubic_spline

# 三次样条曲线生成
class CubicPolynomialWithBoundaryConstrains:
    def __init__(self, init_s, goal_s, init_q, goal_q, init_yaw):
        # 首先记录参数
        self.min_s_ = init_s
        self.max_s_ = goal_s
        delta_s = goal_s - init_s
        assert(delta_s > 0)
        # 计算未知参数
        self.d_ = init_q
        self.c_ = np.tan(init_yaw)
        A = np.array([[delta_s ** 3, delta_s ** 2], [3 * delta_s ** 2, 2 * delta_s]])
        B = np.array([goal_q - self.c_ * delta_s - self.d_, -self.c_]).transpose()
        # 求解中间参量m
        m = np.linalg.solve(A, B)
        # 计算未知参数
        self.a_ = m.transpose()[0]
        self.b_ = m.transpose()[1]
        assert(abs(self.calcValue(goal_s) == goal_q) < common.EPS)

    # 计算三次多项式值
    def calcValue(self, sample):
        # 判断输入范围
        assert(sample >= self.min_s_)
        sample = min(sample, self.max_s_)
        return self.a_ * (sample - self.min_s_) ** 3 + self.b_ * (sample - self.min_s_) ** 2 + self.c_ * (sample - self.min_s_) + self.d_

    # 计算三次多项式一阶导数
    def calcDerivative(self, sample):
        # 判断输入范围
        assert(sample >= self.min_s_)
        sample = min(sample, self.max_s_)
        return 3 * self.a_ * (sample - self.min_s_) ** 2 + 2 * self.b_ * (sample - self.min_s_) + self.c_

    # 计算三次多项式二阶导数
    def calc2Derivative(self, sample):
        # 判断输入范围
        assert(sample >= self.min_s_)
        sample = min(sample, self.max_s_)
        return 6 * self.a_ * (sample - self.min_s_) + 2 * self.b_


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

        # 第一步, 将当前位置转化到frenet坐标系下
        frenet_init_point, init_corresponding_sample =  self.__transPointToFrenet(global_spline, init_point)

        # 第二步,在frenet坐标系下进行规划
        cubic_polynomial = CubicPolynomialWithBoundaryConstrains(frenet_init_point.x_, min(frenet_init_point.x_ + longitude_offset, global_spline.getTotalLength()), frenet_init_point.y_, lateral_offset, frenet_init_point.theta_)

        # 第三步,进行采样,得到路径参数
        sample_number = int((longitude_offset + lateral_offset) / sampling_gap)
        samples = np.linspace(cubic_polynomial.min_s_, cubic_polynomial.max_s_, sample_number)
        frenet_path_s, frenet_path_q, frenet_path_derivative, frenet_path_2derivative = [], [], [], []
        for sample in samples:
            frenet_path_s.append(sample)
            frenet_path_q.append(cubic_polynomial.calcValue(sample))
            frenet_path_derivative.append(cubic_polynomial.calcDerivative(sample))
            frenet_path_2derivative.append(cubic_polynomial.calc2Derivative(sample))

        # 第四步,根据计算出的参数,计算world系下的路径
        world_path_x, world_path_y, world_path_yaw, world_path_curvature = [], [], [], []
        current_sample = init_corresponding_sample
        for i in range(0, len(frenet_path_s)):
            world_cpoint, corresponding_sample = self.__transPointToWorld(global_spline, frenet_path_s[i], frenet_path_q[i], frenet_path_derivative[i], frenet_path_2derivative[i], current_sample)
            # 更新采样点
            current_sample = corresponding_sample
            # 记录world系坐标信息
            world_path_x.append(world_cpoint.x_)
            world_path_y.append(world_cpoint.y_)
            world_path_yaw.append(world_cpoint.theta_)
            world_path_curvature.append(world_cpoint.curvature_)
        local_path = common.CPath(world_path_x, world_path_y, world_path_yaw, world_path_curvature)
        return local_path

    # 根据参数计算world坐标系坐标
    def __transPointToWorld(self, global_spline, s, q, dqs, ddqs, init_sample=0.0):
        # 首先找到当前frenet坐标对应的导航线参考点
        correponding_sample = global_spline.arcLengthToSample(s, init_sample)
        reference_point = global_spline.calcCPoint(correponding_sample)
        arc_length = global_spline.calcArcLength(correponding_sample)
        assert(arc_length is not None)

        # 得到全局导航参考点的参数
        X, Y = global_spline.calcPosition(correponding_sample)
        t = global_spline.calcYaw(correponding_sample)
        k = global_spline.calcKappa(correponding_sample)
        assert(X is not None and X == reference_point.x_)
        assert(Y is not None and Y == reference_point.y_)
        assert(t is not None and t == reference_point.theta_)
        assert(k is not None and k == reference_point.curvature_)
        # 计算world系坐标参数
        offset = s - arc_length
        x = X - q * np.sin(t) + offset * np.cos(t)
        y = Y + q * np.cos(t) + offset * np.sin(t)
        theta = np.arctan(dqs) + t
        S = np.sign(1 - q * k)
        Q = np.sqrt(dqs ** 2 + (1 - q * k) ** 2)
        curvature = S / Q * (k + ((1 - q * k) * ddqs + k * dqs ** 2) / (Q ** 2))
        world_point = common.CPoint(x, y, theta, curvature)
        
        return world_point, correponding_sample


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
        assert(X is not None and X == reference_point.x_)
        assert(Y is not None and Y == reference_point.y_)
        assert(t is not None and t == reference_point.theta_)
        assert(k is not None and k == reference_point.curvature_)

        # 计算Frenet系坐标
        l = arc_length + (x - X) * np.cos(t) + (y - Y) * np.sin(t)
        r = (X - x) * np.sin(t) + (y - Y) * np.cos(t)
        assert(r * k != 1)
        theta = beta - t

        frenet_point = common.CPoint(l, r, theta, 0.0)

        return frenet_point, init_corresponding_sample
        
    
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
    local_path = local_path_factory.generateLocalPath(global_spline, init_point, 40.0)

    # 进行可视化
    fig_1 = plt.figure()
    fig_1_ax = fig_1.add_subplot(1, 1, 1)
    fig_1_ax.axis('equal')
    # 可视化全局导航路径
    global_path_vis, = fig_1_ax.plot(point_x, point_y, ':')
    # 可视化局部路径
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
    # 可视化local_path的曲率随路程的变化曲线
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
    # 可视化local_path的曲率随路程的变化曲线
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