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
import matplotlib.pyplot as plt
import tools.common as common
from collections import Iterable
import collections

class Cycle(common.Navigation):
    def __init__(self, radius, start_angle, end_angle):
        # 验证输入正确性
        assert end_angle > start_angle
        # 保存输入信息
        self.radius_ = radius
        self.start_angle_ = start_angle
        self.end_angle_ = end_angle
        self.total_length_ = self.calcArcLength(self.end_angle_)
    
    # 验证输入正确性
    def verify(self, sample):
        assert sample >= self.start_angle_ and sample <= self.end_angle_
    
    # 计算位置
    def calcPosition(self, samples):
        if isinstance(samples, Iterable):
            x, y = [], []
            for sample in samples:
                self.verify(sample)
                x.append(self.radius_ * np.cos(sample))
                y.append(self.radius_ * np.sin(sample))
            return x, y
        else:
            self.verify(samples)
            x = self.radius_ * np.cos(sample)
            y = self.radius_ * np.sin(sample)
            return x, y
    
     # 计算朝向
    def calcYaw(self, samples):
        if isinstance(samples, Iterable):
            yaws = []
            for sample in samples:
                self.verify(sample)
                yaw = np.arctan2(np.cos(sample), -np.sin(sample))
                yaws.append(yaw)
            return yaws
        else:
            self.verify(samples)
            yaw = np.arctan2(np.cos(samples), -np.sin(samples))
            return yaw
    
    # 计算曲率
    def calcKappa(self, samples):
        if isinstance(samples, Iterable):
            curvatures = []
            for sample in samples:
                self.verify(sample)
                curvatures.append(1.0 / self.radius_)
            return curvatures
        else:
            self.verify(samples)
            curvature = 1.0 / self.radius_
            return curvature
    
    # 计算曲率的变化率
    def calcKappaChangeRate(self, samples):
        if isinstance(samples, Iterable):
            curvature_change_rates = []
            for sample in samples:
                self.verify(sample)
                curvature_change_rates.append(0.0)
            return curvature_change_rates
        else:
            self.verify(samples)
            curvature_change_rate = 0.0
            return curvature_change_rate
    
    # 计算采样点
    def calcCPoint(self, sample):
        x, y = self.calcPosition(sample)
        yaw = self.calcYaw(sample)
        curvature = self.calcKappa(sample)
        cpoint = common.CPoint(x, y, yaw, curvature)
    
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

    plt.show()

if __name__ == "__main__":
    test()