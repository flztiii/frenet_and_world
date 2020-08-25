#! /usr/bin/python3
#! -*- coding:utf-8 -*-

"""

本代码用于生成五次多项式曲线
author: flztiii

"""

import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import tools.common as common
from math import *

# 利用五次多项式生成路径
class QuinticPolynomial:
    def __init__(self, params, eta):
        self.params_ = params
        self.eta_ = eta
        # 生成多项式系数计算
        self.__coefficientGeneration()

    # 生成多项式系数计算
    # x = a0 + a1 * s + a2 * s^2 + a3 * s^3 + a4 * s^4 + a5 * s^5
    # y = b0 + b1 * s + b2 * s^2 + b3 * s^3 + b4 * s^4 + b5 * s^5
    # dx/ds = (dx/dt) / (ds/dt) = Vx / V = cos(theta)
    # ddx/dds = dcos(theta)/ds = (dcos(theta)/dt) / V = -sin(theta) * W / V = -sin(theta) * curvature
    # dy/ds = (dy/dt) / (ds/dt) = Vy / V = sin(theta)
    # ddy/dds = dsin(theta)/ds = (dsin(theta)/dt) / V = cos(theta) * W / V = cos(theta) * curvature    
    def __coefficientGeneration(self):
        a0 = self.params_[0]
        a1 = self.params_[1]
        a2 = self.params_[2] * 0.5
        A = np.array([[self.eta_ ** 3, self.eta_ ** 4, self.eta_ ** 5], [3. * self.eta_ ** 2, 4. * self.eta_ ** 3, 5. * self.eta_ ** 4], [6. * self.eta_, 12. * self.eta_ ** 2, 20. * self.eta_ ** 3]])
        b = np.array([self.params_[3] - a0 - a1 * self.eta_ - a2 * self.eta_ ** 2, self.params_[4] - a1 - 2. * a2 * self.eta_, self.params_[5] - 2. * a2]).T
        x = np.linalg.solve(A, b)
        a3 = x[0]
        a4 = x[1]
        a5 = x[2]
        self.coefficient_ = np.array([a0, a1, a2, a3, a4, a5])

    # 给定输入路程，计算对应值
    def calcValue(self, s):
        return np.dot(self.coefficient_, np.array([1., s, s ** 2, s ** 3, s ** 4, s ** 5]).T)
    
    # 给定输入路程，计算导数
    def calcDerivation(self, s):
        return np.dot(self.coefficient_[1:], np.array([1., 2. * s, 3. * s ** 2, 4. * s ** 3, 5. * s ** 4]))
    
    # 给定输入路程，计算二阶导数
    def calc2Derivation(self, s):
        return np.dot(self.coefficient_[2:], np.array([2. , 6. * s, 12. * s ** 2, 20. * s ** 3]))
    
    # 给定输入路程，计算三阶导数
    def calc3Derivation(self, s):
        return np.dot(self.coefficient_[3:], np.array([6., 24. * s, 60. * s ** 2]))   

# G2 曲线
class G2Spline:
    def __init__(self, point_start, point_goal):
        # 判断输入是否正确
        assert isinstance(point_start, common.CPoint) and isinstance(point_goal, common.CPoint)
        # 构建输入参数
        params_x = [point_start.x_, cos(point_start.theta_), - sin(point_start.theta_) * point_start.curvature_, point_goal.x_, cos(point_goal.theta_), - sin(point_goal.theta_) * point_goal.curvature_]
        params_y = [point_start.y_, sin(point_start.theta_), cos(point_start.theta_) * point_start.curvature_, point_goal.y_, sin(point_goal.theta_), cos(point_goal.theta_) * point_goal.curvature_]
        # 构建参数eta
        self.eta_ = np.linalg.norm(np.array([point_goal.x_, point_goal.y_]) - np.array([point_start.x_, point_start.y_]))
        # 构建曲线函数
        self.__spline_x_ = QuinticPolynomial(params_x, self.eta_)
        self.__spline_y_ = QuinticPolynomial(params_y, self.eta_)
    
    # 输入路程，计算坐标
    def calcPosition(self, sample):
        return self.__spline_x_.calcValue(sample), self.__spline_y_.calcValue(sample)
    
    # 输入路程，计算朝向
    def calcYaw(self, sample):
        dx = self.__spline_x_.calcDerivation(sample)
        dy = self.__spline_y_.calcDerivation(sample)
        return common.pi_2_pi(atan2(dy, dx))
    
    # 输入路程，计算曲率
    def calcCurvature(self, sample):
        dx = self.__spline_x_.calcDerivation(sample)
        dy = self.__spline_y_.calcDerivation(sample)
        ddx = self.__spline_x_.calc2Derivation(sample)
        ddy = self.__spline_y_.calc2Derivation(sample)
        curvature = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return curvature

# 测试函数
def test():
    # 设定初始点
    init_point = common.CPoint(0, 0, 2.80419, 0.0)
    goal_point = common.CPoint(10, -2.86393, -0.226702, 0.0)
    g2_spline = G2Spline(init_point, goal_point)
    # 进行采样
    gap = 0.1
    samples = np.arange(0.0, g2_spline.eta_ + gap, gap)
    points_x, points_y, points_yaw, points_curvature = [], [], [], []
    for sample in samples:
        x, y = g2_spline.calcPosition(sample)
        yaw = g2_spline.calcYaw(sample)
        curvature = g2_spline.calcCurvature(sample)
        points_x.append(x)
        points_y.append(y)
        points_yaw.append(yaw)
        points_curvature.append(curvature)
    # 可视化
    fig = plt.figure(figsize=(14, 14))
    # 可视化行驶过程
    ax1 = fig.add_subplot(3, 1, 1)
    # 绘制曲线
    ax1.plot(points_x, points_y, ":")
    # for i in range(0, len(samples)):
    #     ax1.arrow(points_x[i], points_y[i], cos(points_yaw[i]), sin(points_yaw[i]), fc='r', ec='k', head_width=0.5, head_length=0.5)
    # 显示范围
    AREA = 5
    ax1.set_xlim(init_point.x_ - AREA, goal_point.x_ + AREA)
    ax1.set_ylim(init_point.y_ - AREA, goal_point.y_ + AREA)
    ax1.set_aspect(1)
    # 显示网格
    ax1.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
    # 设置x,y轴名称
    ax1.set_xlabel('x position[m]')
    ax1.set_ylabel('y position[m]')

    # 可视化朝向随路程变化曲线
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(samples, points_yaw)
    # 显示网格
    ax2.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
    # 设置x,y轴名称
    ax2.set_xlabel('distance[m]')
    ax2.set_ylabel('yaw[rad]')

    # 可视化曲率随路程变化曲线
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(samples, points_curvature)
    # 显示网格
    ax3.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
    # 设置x,y轴名称
    ax3.set_xlabel('distance[m]')
    ax3.set_ylabel('curvature[rad/m]')

    # 设置第二个初始点
    init_point = common.CPoint(0.0, 0.0, 0.0, 0.25)
    goal_point = common.CPoint(10.0, 2.0, 1.56, 0.25)
    g2_spline = G2Spline(init_point, goal_point)
    # 进行采样
    gap = 0.1
    samples = np.arange(0.0, g2_spline.eta_ + gap, gap)
    points_x, points_y, points_yaw, points_curvature = [], [], [], []
    for sample in samples:
        x, y = g2_spline.calcPosition(sample)
        yaw = g2_spline.calcYaw(sample)
        curvature = g2_spline.calcCurvature(sample)
        points_x.append(x)
        points_y.append(y)
        points_yaw.append(yaw)
        points_curvature.append(curvature)
    # 可视化
    fig = plt.figure(figsize=(14, 14))
    # 可视化行驶过程
    ax1 = fig.add_subplot(3, 1, 1)
    # 绘制曲线
    ax1.plot(points_x, points_y, ":")
    # for i in range(0, len(samples)):
    #     ax1.arrow(points_x[i], points_y[i], cos(points_yaw[i]), sin(points_yaw[i]), fc='r', ec='k', head_width=0.5, head_length=0.5)
    # 显示范围
    AREA = 5
    ax1.set_xlim(init_point.x_ - AREA, goal_point.x_ + AREA)
    ax1.set_ylim(init_point.y_ - AREA, goal_point.y_ + AREA)
    ax1.set_aspect(1)
    # 显示网格
    ax1.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
    # 设置x,y轴名称
    ax1.set_xlabel('x position[m]')
    ax1.set_ylabel('y position[m]')

    # 可视化朝向随路程变化曲线
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(samples, points_yaw)
    # 显示网格
    ax2.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
    # 设置x,y轴名称
    ax2.set_xlabel('distance[m]')
    ax2.set_ylabel('yaw[rad]')

    # 可视化曲率随路程变化曲线
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(samples, points_curvature)
    # 显示网格
    ax3.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
    # 设置x,y轴名称
    ax3.set_xlabel('distance[m]')
    ax3.set_ylabel('curvature[rad/m]')

    plt.show()

if __name__ == "__main__":
    test()