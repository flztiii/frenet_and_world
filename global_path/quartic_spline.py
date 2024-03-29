#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

本代码利用四次样条生成全局导航路径
author: flztiii

"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import math
import matplotlib.pyplot as plt
import bisect
import tools.common as common
from scipy import integrate
from collections import Iterable
import collections

# 构建四次样条曲线
class Spline:
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x_ = x
        self.y_ = y
        # 散点数量
        self.number_ = len(x)
        # 最大下标(也是方程式的个数，参数个数)
        self.n_ = len(x) - 1
        # 中间参数
        self.h_ = np.diff(x)
        self.delta_ = []
        for i in range(0, self.n_):
            self.delta_.append((self.y_[i + 1] - self.y_[i]) / self.h_[i])
        self.delta_ = np.array(self.delta_)
        self.chi_ = []
        for i in range(1, self.n_):
            self.chi_.append(self.h_[i] + self.h_[i - 1])
        self.chi_ = np.array(self.chi_)

        # 计算矩形
        A = np.zeros((self.n_ - 1, self.n_ - 1))
        for i in range(0, self.n_ - 1):
            for j in range(0, self.n_ - 1):
                if j < i - 1:
                    A[i, j] = 0
                elif j == i - 1:
                    A[i, j] = self.h_[i] ** 2 / 3.0
                elif j == i:
                    A[i, j] =  self.chi_[i] * self.chi_[j]
                elif j == i + 1:
                    A[i, j] = 2.0 * self.chi_[i] * self.chi_[j] - self.h_[i + 1] ** 2 / 3.0
                else:
                    A[i, j] = 2.0 * self.chi_[i] * self.chi_[j]
        B = np.zeros(self.n_ - 1)
        for i in range(0, self.n_ - 1):
            B[i] = 8.0 * (self.delta_[i] - self.delta_[i + 1])
        result = np.linalg.solve(A, B)
        self.z_ = np.zeros(self.n_ + 1)
        for i in range(1, self.n_):
            self.z_[i] = result[i - 1]

        # 计算剩余参量
        self.c_ = np.zeros(self.n_)
        self.d_ = np.zeros(self.n_)
        self.e_ = np.zeros(self.n_)
        for i in range(0, self.n_):
            self.d_[i] = - self.z_[i + 1] / 24.0 * self.h_[i] ** 2 + self.y_[i + 1] / self.h_[i]
            self.e_[i] = self.z_[i] / 24.0 * self.h_[i] ** 2 + self.y_[i] / self.h_[i]
        for i in list(reversed(range(1, self.n_))):
            self.c_[i - 1] = self.c_[i] + (self.h_[i] + self.h_[i - 1]) / 4.0 * self.z_[i]
    
    # 计算对应函数段
    def __getSegment(self, sample):
        return min(bisect.bisect(self.x_, sample) - 1, self.n_ - 1)

    # 计算位置
    def calc(self, sample):
        if sample < self.x_[0] or sample > self.x_[-1]:
            return None
        index = self.__getSegment(sample)
        return self.z_[index + 1] / (24.0 * self.h_[index]) * (sample - self.x_[index]) ** 4 - self.z_[index] / (24.0 * self.h_[index]) * (self.x_[index + 1] - sample) ** 4 + self.c_[index] * (sample - self.x_[index]) * (self.x_[index + 1] - sample) + self.d_[index] * (sample - self.x_[index]) + self.e_[index] * (self.x_[index + 1] - sample)
    
     # 计算一阶导数
    def calcd(self, sample):
        if sample < self.x_[0] or sample > self.x_[-1]:
            return None
        index = self.__getSegment(sample)
        return self.z_[index + 1] / (6.0 * self.h_[index]) * (sample - self.x_[index]) ** 3 + self.z_[index] / (6.0 * self.h_[index]) * (self.x_[index + 1] - sample) ** 3 + self.c_[index] * (self.x_[index] + self.x_[index + 1] - 2.0 * sample) + self.d_[index] - self.e_[index]

    # 计算二阶导数
    def calcdd(self, sample):
        if sample < self.x_[0] or sample > self.x_[-1]:
            return None
        index = self.__getSegment(sample)
        return self.z_[index + 1] / (2.0 * self.h_[index]) * (sample - self.x_[index]) ** 2 - self.z_[index] / (2.0 * self.h_[index]) * (self.x_[index + 1] - sample) ** 2 - 2.0 * self.c_[index]

    # 计算三阶导数
    def calcddd(self, sample):
        if sample < self.x_[0] or sample > self.x_[-1]:
            return None
        index = self.__getSegment(sample)
        return self.z_[index + 1] / self.h_[index] * (sample - self.x_[index]) + self.z_[index] / self.h_[index] * (self.x_[index + 1] - sample)

# 构建2d四次样条曲线
class QuarticSpline2D(common.Navigation):
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x_ = x
        self.y_ = y
        # 计算路程
        dx = np.diff(x)
        dy = np.diff(y)
        ds = [math.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]
        self.s_ = [0.0]
        self.s_.extend(np.cumsum(ds))
        # 开始构建曲线
        self.generateSpline()
        # 生成里程查询表
        self.arc_length_checking_table_ = self.__generateArcLengthCheckingTable()
        # 得到总弧长
        self.total_length_ = self.calcArcLength(self.s_[-1])
    
    # 得到采样的上限
    def maxSample(self):
        return self.s_[-1]
    
    # 得到采样的下限
    def minSample(self):
        return self.s_[0]
    
    # 获取曲线x坐标属性
    def getXAttribute(self):
        return self.spline_x_
    
    # 获取曲线y坐标信息
    def getYAttribute(self):
        return self.spline_y_
    
    # 构建曲线
    def generateSpline(self):
        self.spline_x_ = Spline(self.s_, self.x_)
        self.spline_y_ = Spline(self.s_, self.y_)
    
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


    # 计算里程微元
    def calcArcLengthDerivative(self, sample):
        return max(np.linalg.norm(np.array([self.spline_x_.calcd(sample), self.spline_y_.calcd(sample)])), 1e-6)
    
    # 计算里程
    def calcArcLength(self, sample):
        # 判断查询字典是否存在
        if hasattr(self, 'arc_length_checking_table_'):
            # 查询字典存在
            # 首先对字典进行查询
            init_index = int(sample / 0.1)
            while list(self.arc_length_checking_table_.keys())[init_index] >= sample:
                if init_index == 0:
                    break
                else:
                    init_index -= 1
            last_sample = list(self.arc_length_checking_table_.keys())[init_index]
            arc_length = self.arc_length_checking_table_[last_sample] + integrate.quad(self.calcArcLengthDerivative, last_sample, sample)[0]
        else:
            # 查询字典不存在
            arc_length = integrate.quad(self.calcArcLengthDerivative, 0.0, sample)[0]
        return arc_length

    # 生成路程查询表
    def __generateArcLengthCheckingTable(self):
        arc_length_checking_table = collections.OrderedDict()
        gap = 0.1
        sample_number = int((self.s_[-1] - self.s_[0]) / gap) + 1
        samples = np.linspace(self.s_[0], self.s_[-1] - common.EPS, sample_number)
        for sample in samples:
            arc_length = self.calcArcLength(sample)
            arc_length_checking_table[sample] = arc_length
        return arc_length_checking_table
    
    # 给出里程计算对应采样点
    def arcLengthToSample(self, arc_length, init_sample = 0.0):
        assert arc_length < self.total_length_
        sample = init_sample
        while (sample >= self.s_[0] and sample <= self.s_[-1]) and np.abs(arc_length - self.calcArcLength(sample)) > 1e-3:
            sample += (arc_length - self.calcArcLength(sample)) / self.calcArcLengthDerivative(sample)
        sample = max(self.s_[0], min(sample, self.s_[-1]))
        return sample

    # 给出最长里程
    def getTotalLength(self):
        return self.total_length_ - common.EPS

# 测试四次插值函数
def test():
    x = [0.0, 10.0, 20.0, 30.0]
    y = [0.0, 1.0, 5.0, 25.0]
    spline = Spline(x,y)
    samples = np.linspace(0.0, 30.0, 600)
    results,results_d, results_dd, results_ddd = [], [], [], []
    for sample in samples:
        results.append(spline.calc(sample))
        results_d.append(spline.calcd(sample))
        results_dd.append(spline.calcdd(sample))
        results_ddd.append(spline.calcddd(sample))
    
    # 进行可视化
    fig1 = plt.figure()
    plt.plot(samples, results)
    plt.plot(x, y, 'xk')

    fig2 = plt.figure()
    plt.plot(samples, results_d)

    fig3 = plt.figure()
    plt.plot(samples, results_dd)

    fig4 = plt.figure()
    plt.plot(samples, results_ddd)

    plt.show()


# 测试四次样条函数
def test2():
    # 初始化散点
    x = [0.0, 20.0, 0.0]
    y = [0.0, 20.0, 40.0]
    # 初始化采样间隔
    gap = 0.1
    # 构建2d四次样条曲线
    quartic_spline = QuarticSpline2D(x, y)
    # 对2d四次样条曲线进行采样
    sample_s = np.arange(0.0, quartic_spline.s_[-1], gap)
    point_x, point_y = quartic_spline.calcPosition(sample_s)
    point_yaw = quartic_spline.calcYaw(sample_s)
    point_kappa = quartic_spline.calcKappa(sample_s)
    point_kappa_change_rate = quartic_spline.calcKappaChangeRate(sample_s)
    estimate_distance = 0.0
    for i in range(0, len(point_x) - 1):
        estimate_distance += np.sqrt((point_x[i + 1] - point_x[i]) ** 2 + (point_y[i + 1] - point_y[i]) ** 2)
    distance = quartic_spline.calcArcLength(sample_s[-1])
    print("estimate_distance ", estimate_distance, ", distance ", distance)
    expected_sample = quartic_spline.arcLengthToSample(4.0)
    print("expected_sample ", expected_sample)
    verify_samples = np.linspace(0.0, expected_sample + 10e-10, 200)
    verify_point_x, verify_point_y = quartic_spline.calcPosition(verify_samples)
    verify_distance = 0.0
    for i in range(0, len(verify_point_x) - 1):
        verify_distance += np.sqrt((verify_point_x[i + 1] - verify_point_x[i]) ** 2 + (verify_point_y[i + 1] - verify_point_y[i]) ** 2)
    print("verify_distance ", verify_distance)

    # 进行可视化
    plt.subplots(1)
    plt.plot(x, y, "xb", label="input")
    plt.plot(point_x, point_y, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.subplots(1)
    plt.plot(sample_s, [np.rad2deg(iyaw) for iyaw in point_yaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    plt.subplots(1)
    plt.plot(sample_s, point_kappa, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [rad/m]")

    plt.subplots(1)
    plt.plot(sample_s, point_kappa_change_rate, "-r", label="curvature change rate")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature change rate [rad/m^2]")

    plt.show()

if __name__ == "__main__":
    test2()