#! /usr/bin/python3
#! -*- coding:utf-8 -*-

"""

通用工具模块
author: flztiii

"""

from math import *
import numpy as np
import copy
import abc

EPS = 1e-8

# 点
class Point(object):
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y

    def __sub__(self, point):
        return Vector(self.x_ - point.x_, self.y_ - point.y_)

# 向量
class Vector:
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y
    
    # 重载加法
    def __add__(self, vector):
        return Vector(self.x_ + vector.x_, self.y_ + vector.y_)
    
    # 重载减法
    def __sub__(self, vector):
        return Vector(self.x_ - vector.x_, self.y_ - vector.y_)

    # 获取向量唱的
    def value(self):
        return sqrt(self.x_ ** 2 + self.y_ ** 2)


# 曲率点
class CPoint(Point):
    def __init__(self, x, y, theta, curvature):
        super(CPoint, self).__init__(x, y)
        self.theta_ = theta
        self.curvature_ = curvature

# 速度点
class VPoint(CPoint):
    def __init__(self, x, y, theta, curvature, velocity):
        super(VPoint, self).__init__(x, y, theta, curvature)
        self.velocity_ = velocity

# 曲率点路径
class CPath(object):
    def __init__(self, points_x, points_y, points_yaw, points_curvature):
        self.points_x_ = copy.deepcopy(points_x)  # x坐标
        self.points_y_ = copy.deepcopy(points_y)  # y坐标
        self.points_yaw_ = copy.deepcopy(points_yaw)  # 朝向
        self.points_curvature_ = copy.deepcopy(points_curvature)  # 曲率
        # 生成路径
        self.__generatePath()
    
    # 生成路径
    def __generatePath(self):
        self.path_ = []  # 路径
        self.points_dis_ = []  # 路程
        distance = 0.0
        for i in range(0, len(self.points_x_)):
            # 添加路径点
            point = CPoint(self.points_x_[i], self.points_y_[i], self.points_yaw_[i], self.points_curvature_[i])
            self.path_.append(point)
            # 路径点的路程
            self.points_dis_.append(distance)
            if i < len(self.points_x_) - 1:
                distance += np.linalg.norm(np.array([self.points_x_[i + 1], self.points_y_[i + 1]]) - np.array([self.points_x_[i], self.points_y_[i]]))

# 轨迹
class Trajectory:
    def __init__(self, cpath, velocitys, accelerations, jerks):
        self.points_x_ = copy.deepcopy(cpath.points_x_)  # x坐标
        self.points_y_ = copy.deepcopy(cpath.points_y_)  # y坐标
        self.points_yaw_ = copy.deepcopy(cpath.points_yaw_)  # 朝向
        self.points_curvature_ = copy.deepcopy(cpath.points_curvature_)  # 曲率
        self.path_ = copy.deepcopy(cpath.path_)  # 路径
        self.points_dis_ = copy.deepcopy(cpath.points_dis_)  # 路程
        self.velocitys_ = copy.deepcopy(velocitys)  # 速度
        self.accelerations_ = copy.deepcopy(accelerations)  # 加速度
        self.jerks_ = copy.deepcopy(jerks)  # 冲击

# 全局导航
class Navigation(abc.ABC):
    # 计算位置
    @abc.abstractmethod
    def calcPosition(self, samples):
        return NotImplemented
    
    # 计算朝向
    @abc.abstractmethod
    def calcYaw(self, samples):
        return NotImplemented

    # 计算曲率
    @abc.abstractmethod
    def calcKappa(self, samples):
        return NotImplemented

    # 计算曲率的变化率
    @abc.abstractmethod
    def calcKappaChangeRate(self, samples):
        return NotImplemented
    
    # 计算采样点
    @abc.abstractmethod
    def calcCPoint(self, sample):
        return NotImplemented

    # 计算里程
    @abc.abstractmethod
    def calcArcLength(self, sample):
        return NotImplemented
    
    # 给出里程计算对应采样点
    @abc.abstractmethod
    def arcLengthToSample(self, arc_length, init_sample = 0.0):
        return NotImplemented
    
    # 给出最长里程
    @abc.abstractmethod
    def getTotalLength(self):
        return NotImplemented

# 保证弧度在-pi~pi之间
def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi

# 高斯函数
def gaussian(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

# 坐标转化，将世界坐标系下的点转换到另一个坐标系下
def coordinateTransform(x, y, coordinate):
    # coordinate可以是曲率点也可以是速度点
    delta_x = x - coordinate.x_
    delta_y = y - coordinate.y_
    new_x = delta_x * cos(coordinate.theta_) + delta_y * sin(coordinate.theta_)
    new_y = - delta_x * sin(coordinate.theta_) + delta_y * cos(coordinate.theta_)
    return new_x, new_y

# 坐标转化，将另一个坐标系下的点转换到世界坐标系下
def coordinateInvTransform(x, y, coordinate):
    new_x = coordinate.x_ + x * cos(coordinate.theta_) - y * sin(coordinate.theta_)
    new_y = coordinate.y_ + x * sin(coordinate.theta_) + y * cos(coordinate.theta_)
    # new_x = coordinate.x_ - y * sin(coordinate.theta_)
    # new_y = coordinate.y_ + y * cos(coordinate.theta_)
    return new_x, new_y

# 计算给出点在路径的哪一个点附近
def getIndexofPath(x, y, path, last_index = 0):
    # path可以是CPoint的list也可以是VPoint的list
    index = None
    for i in range(last_index, len(path) - 1):
        x_offset_before, _ = coordinateTransform(x, y, path[i])
        x_offset_after, _ = coordinateTransform(x, y, path[i + 1])
        if x_offset_before >= 0.0 and x_offset_after <= 0.0:
            if abs(x_offset_before) <= abs(x_offset_after):
                index = i
            else:
                index = i + 1
    return index;

if __name__ == "__main__":
    pass