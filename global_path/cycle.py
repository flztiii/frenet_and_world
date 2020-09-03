#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

本代码利用圆生成全局导航路径
author: flztiii

"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import math
import matplotlib.pyplot as plt
import tools.common as common
from collections import Iterable
import collections


class AttributeX:
    def __init__(self, radius, start_angle, end_angle):
         # 验证输入正确性
        assert end_angle > start_angle
        # 保存输入信息
        self.radius_ = radius
        self.start_angle_ = start_angle
        self.end_angle_ = end_angle

    # 验证输入正确性
    def verify(self, sample):
        assert sample >= self.start_angle_ and sample <= self.end_angle_

    def calc(self, sample):
        self.verify(sample)
        return self.radius_ * np.cos(sample)

    def calcd(self, sample):
        self.verify(sample)
        return - self.radius_ * np.sin(sample)

    def calcdd(self, sample):
        self.verify(sample)
        return - self.radius_ * np.cos(sample)

    def calcddd(self, sample):
        self.verify(sample)
        return self.radius_ * np.sin(sample)

class AttributeY:
    def __init__(self, radius, start_angle, end_angle):
         # 验证输入正确性
        assert end_angle > start_angle
        # 保存输入信息
        self.radius_ = radius
        self.start_angle_ = start_angle
        self.end_angle_ = end_angle

    # 验证输入正确性
    def verify(self, sample):
        assert sample >= self.start_angle_ and sample <= self.end_angle_

    def calc(self, sample):
        self.verify(sample)
        return self.radius_ * np.sin(sample)

    def calcd(self, sample):
        self.verify(sample)
        return self.radius_ * np.cos(sample)

    def calcdd(self, sample):
        self.verify(sample)
        return - self.radius_ * np.sin(sample)

    def calcddd(self, sample):
        self.verify(sample)
        return - self.radius_ * np.cos(sample)

class Cycle(common.Navigation):
    def __init__(self, radius, start_angle, end_angle):
        # 验证输入正确性
        assert end_angle > start_angle
        # 保存输入信息
        self.radius_ = radius
        self.start_angle_ = start_angle
        self.end_angle_ = end_angle
        # 得到x和y坐标表达
        self.generateSpline()
        self.total_length_ = self.calcArcLength(self.end_angle_)
    
    # 得到采样的上限
    def maxSample(self):
        return self.end_angle_
    
    # 得到采样的下限
    def minSample(self):
        return self.start_angle_

    # 构建曲线
    def generateSpline(self):
        self.spline_x_ = AttributeX(self.radius_, self.start_angle_, self.end_angle_)
        self.spline_y_ = AttributeY(self.radius_, self.start_angle_, self.end_angle_)
    
    # 获取曲线x坐标属性
    def getXAttribute(self):
        return self.spline_x_
    
    # 获取曲线y坐标信息
    def getYAttribute(self):
        return self.spline_y_

    # 验证输入正确性
    def verify(self, sample):
        assert sample >= self.start_angle_ and sample <= self.end_angle_

    # 计算位置
    def calcPosition(self, samples):
        if isinstance(samples, Iterable):
            x, y = [], []
            for sample in samples:
                if not self.spline_x_.calc(sample) is None:
                    x.append(self.spline_x_.calc(sample))
                    y.append(self.spline_y_.calc(sample))
            return x, y
        else:
            sample = samples
            if not self.spline_x_.calc(sample) is None:
                x = self.spline_x_.calc(sample)
                y = self.spline_y_.calc(sample)
                return x, y
            else:
                return None, None
    
    # 计算朝向
    def calcYaw(self, samples):
        if isinstance(samples, Iterable):
            yaws = []
            for sample in samples:
                dx = self.spline_x_.calcd(sample)
                dy = self.spline_y_.calcd(sample)
                if not dx is None:
                    yaws.append(math.atan2(dy, dx))
            return yaws
        else:
            sample = samples
            dx = self.spline_x_.calcd(sample)
            dy = self.spline_y_.calcd(sample)
            if not dx is None:
                yaw = math.atan2(dy, dx)
                return yaw
            else:
                return None

    # 计算曲率
    def calcKappa(self, samples):
        if isinstance(samples, Iterable):
            kappas = []
            for sample in samples:
                dx = self.spline_x_.calcd(sample)
                dy = self.spline_y_.calcd(sample)
                ddx = self.spline_x_.calcdd(sample)
                ddy = self.spline_y_.calcdd(sample)
                if not dx is None:
                    kappa = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
                    kappas.append(kappa)
            return kappas
        else:
            sample = samples
            dx = self.spline_x_.calcd(sample)
            dy = self.spline_y_.calcd(sample)
            ddx = self.spline_x_.calcdd(sample)
            ddy = self.spline_y_.calcdd(sample)
            if not dx is None:
                kappa = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
                return kappa
            else:
                return None
    
    # 计算曲率的变化率
    def calcKappaChangeRate(self, samples):
        if isinstance(samples, Iterable):
            kappa_change_rates = []
            for sample in samples:
                dx = self.spline_x_.calcd(sample)
                dy = self.spline_y_.calcd(sample)
                ddx = self.spline_x_.calcdd(sample)
                ddy = self.spline_y_.calcdd(sample)
                dddx = self.spline_x_.calcddd(sample)
                dddy = self.spline_y_.calcddd(sample)
                if not dx is None:
                    kappa_change_rate = (6.0 * (dy * ddx - dx * ddy) * (dx * ddx + dy * ddy) + 2.0 * (np.power(dx, 2) + np.power(dy, 2)) * (-dy * dddx + dx * dddy)) / (2.0 * np.power(np.power(dx, 2) + np.power(dy, 2), 2.5))
                    kappa_change_rates.append(kappa_change_rate)
            return kappa_change_rates
        else:
            sample = samples
            dx = self.spline_x_.calcd(sample)
            dy = self.spline_y_.calcd(sample)
            ddx = self.spline_x_.calcdd(sample)
            ddy = self.spline_y_.calcdd(sample)
            dddx = self.spline_x_.calcddd(sample)
            dddy = self.spline_y_.calcddd(sample)
            if not dx is None:
                kappa_change_rate = (6.0 * (dy * ddx - dx * ddy) * (dx * ddx + dy * ddy) + 2.0 * (np.power(dx, 2) + np.power(dy, 2)) * (-dy * dddx + dx * dddy)) / (2.0 * np.power(np.power(dx, 2) + np.power(dy, 2), 2.5))
                return kappa_change_rate
            else:
                return None
    
    # 计算采样点
    def calcCPoint(self, sample):
        if not self.spline_x_.calc(sample) is None:
            x = self.spline_x_.calc(sample)
            y = self.spline_y_.calc(sample)
            dx = self.spline_x_.calcd(sample)
            dy = self.spline_y_.calcd(sample)
            ddx = self.spline_x_.calcdd(sample)
            ddy = self.spline_y_.calcdd(sample)
            yaw = math.atan2(dy, dx)
            kappa = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
            point = common.CPoint(x, y, yaw, kappa)
            return point
        else:
            return None
    
    # 计算里程
    def calcArcLength(self, sample):
        self.verify(sample)
        return (sample - self.start_angle_) * self.radius_

    # 给出里程计算对应采样点
    def arcLengthToSample(self, arc_length, init_sample = 0.0):
        assert arc_length <= self.total_length_
        return arc_length / self.radius_ + self.start_angle_
    
    # 给出最长里程
    def getTotalLength(self):
        return self.total_length_ - common.EPS

# 测试函数
def test():
    # 构建圆
    cycle = Cycle(20.0, -np.pi/2, np.pi/2)
    # 进行采样
    samples = np.linspace(-np.pi/2, np.pi/2 - common.EPS, 200)
    points_x, points_y = cycle.calcPosition(samples)
    points_yaw = cycle.calcYaw(samples)
    points_curvature = cycle.calcKappa(samples)
    points_curvature_change_rate = cycle.calcKappaChangeRate(samples)
    points_dis = []
    for sample in samples:
        points_dis.append(cycle.calcArcLength(sample))
    
    # 进行可视化
    # 可视化路径
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis('equal')
    ax1.plot(points_x, points_y)

    # 可视化朝向随里程变化
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(points_dis, points_yaw)

    # 可视化曲率随里程变化
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.plot(points_dis, points_curvature)

    # 可视化曲率变化率随里程变化
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.plot(points_dis, points_curvature_change_rate)

    plt.show()

if __name__ == "__main__":
    test()