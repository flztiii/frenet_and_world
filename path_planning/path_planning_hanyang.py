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
        assert(self.calcValue(goal_s) == goal_q)

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
        assert(isinstance(init_point, common.CPoint))


# 测试函数
def test():
    cubic_polynomial = CubicPolynomialWithBoundaryConstrains(10.0, 20.0, 0.5, 3.0, 0.3)

if __name__ == "__main__":
    test()