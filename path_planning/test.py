#! /usr/bin/python3
#! -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from math import *
import tools.common as common
import path_planning.g2_spline as g2_spline
import global_path.cubic_spline as cubic_spline

# 将frenet坐标转到world系下
def transPointToWorld(point, global_path, reference_index):
    # 首先得到frenet坐标参数
    l = point.x_
    r = point.y_
    beta = point.theta_
    alpha = point.curvature_

    # 得到全局导航参考点的参数
    X = global_path.points_x_[reference_index]
    Y = global_path.points_y_[reference_index]
    t = global_path.points_yaw_[reference_index]
    k = global_path.points_curvature_[reference_index]
    m = 0.0
    # 计算m=dk/ds, 曲率的导数
    if reference_index > 0:
        ds = global_path.points_dis_[reference_index] - global_path.points_dis_[reference_index - 1]
        dk = global_path.points_curvature_[reference_index] - global_path.points_curvature_[reference_index - 1]
        assert ds != 0
        m = dk / ds
    else:
        ds = global_path.points_dis_[reference_index + 1] - global_path.points_dis_[reference_index]
        dk = global_path.points_curvature_[reference_index + 1] - global_path.points_curvature_[reference_index]
        assert ds != 0
        m = dk / ds

    # 计算world系坐标参数
    x = X - r * np.sin(t)
    y = Y + r * np.cos(t)
    theta = np.arctan2(np.cos(t)*np.sin(beta)+(1 - r*k)*np.cos(beta)*np.sin(t), (1 - r*k)*np.cos(beta)*np.cos(t)-np.sin(beta)*np.sin(t))
    curvature = (alpha - r * alpha * k + k * (1 - r * k)**2 * np.cos(beta)**3 + m * r * np.cos(beta)**2 * np.sin(beta) + 2 * k * np.cos(beta) * np.sin(beta)**2) / np.power((1 - r * k)**2 * np.cos(beta)**2 + np.sin(beta)**2, 1.5)

    world_point = common.CPoint(x, y, theta, curvature)
    return world_point

# 将world系坐标转化到frenet系下
def transPointToFrenet(point, global_path, reference_index):
    # 首先得到world系坐标参数
    x = point.x_
    y = point.y_
    beta = point.theta_
    alpha = point.curvature_

    # 得到全局导航参考点的参数
    X = global_path.points_x_[reference_index]
    Y = global_path.points_y_[reference_index]
    t = global_path.points_yaw_[reference_index]
    k = global_path.points_curvature_[reference_index]
    m = 0.0
    # 计算m=dk/ds, 曲率的导数
    if reference_index > 0:
        ds = global_path.points_dis_[reference_index] - global_path.points_dis_[reference_index - 1]
        dk = global_path.points_curvature_[reference_index] - global_path.points_curvature_[reference_index - 1]
        assert ds != 0
        m = dk / ds
    else:
        ds = global_path.points_dis_[reference_index + 1] - global_path.points_dis_[reference_index]
        dk = global_path.points_curvature_[reference_index + 1] - global_path.points_curvature_[reference_index]
        assert ds != 0
        m = dk / ds
    
    # 计算Frenet系坐标参数
    l = global_path.points_dis_[reference_index]
    r = (X - x) * np.sin(t) + (y - Y) * np.cos(t)
    assert(r * k != 1)
    theta = np.arctan2(np.sin(beta - t), np.cos(beta - t) / (1 - r * k))
    Q = -2 + r * k * (2 - r * k) + r * k * (-2 + r * k) * np.cos(2 * beta - 2 * t)
    curvature = np.sign(1 - r * k) * (-4 * alpha * (1 - r * k)**2 + k * (-1 + r * k) * (np.cos(3 * beta - 3 * t) - 5 * np.cos(beta - t)) + 4 * m * r * np.cos(beta - t) ** 2 * np.sin(beta - t)) / (sqrt(-2 * Q) * Q)

    frenet_point = common.CPoint(l, r, theta, curvature)
    return frenet_point

# 测试函数1, 测试frenet到world
def test1():
    # 首先构建一个圆
    point_x, point_y, point_yaw, point_kappa = [], [], [], []
    samples = np.linspace(-np.pi, np.pi, 10000)
    radius = 20.0
    for sample in samples:
        point_x.append(radius * np.cos(sample))
        point_y.append(radius * np.sin(sample))
        point_yaw.append(np.arctan2(np.cos(sample), -np.sin(sample)))
        point_kappa.append(1.0 / radius)
    # 构建全局导航路径
    global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)

    # 第二步,给出在frenet系的坐标
    corresponding_index = 0
    print("sample ", samples[corresponding_index])
    corresponding_reference_point = global_path.path_[corresponding_index]
    init_frenet_point = common.CPoint(global_path.points_dis_[corresponding_index], 30.0, 3.1415, 0.0)
    # 进行frenet到world的坐标转化
    init_world_point = transPointToWorld(init_frenet_point, global_path, corresponding_index)
    print("init_world_point: ", init_world_point.x_, ", ", init_world_point.y_, ", ", init_world_point.theta_, ", ", init_world_point.curvature_)

    # 进行可视化
    plt.figure()
    plt.axis('equal')
    # 可视化全局导航路径
    plt.plot(point_x, point_y, ':')
    plt.arrow(corresponding_reference_point.x_, corresponding_reference_point.y_, radius * np.cos(corresponding_reference_point.theta_), radius * np.sin(corresponding_reference_point.theta_), fc='r', ec='k', head_width=0.5, head_length=0.5)
    # 可视化点
    plt.arrow(init_world_point.x_, init_world_point.y_, np.cos(init_world_point.theta_), np.sin(init_world_point.theta_), fc='b', ec='k', head_width=0.5, head_length=0.5)
    plt.show()

# 测试函数2,测试world到frenet
def test2():
    # 首先构建一个圆
    point_x, point_y, point_yaw, point_kappa = [], [], [], []
    samples = np.linspace(0, np.pi, 10000)
    radius = 20.0
    for sample in samples:
        point_x.append(radius * np.cos(sample))
        point_y.append(radius * np.sin(sample))
        point_yaw.append(np.arctan2(np.cos(sample), -np.sin(sample)))
        point_kappa.append(1.0 / radius)
    # 构建全局导航路径
    global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)

    # 第二步,给出在world系的坐标
    for d in range(-15,20,5):
        corresponding_index = 0
        print("sample ", samples[corresponding_index])
        corresponding_reference_point = global_path.path_[corresponding_index]
        offset = d
        print("offset", d)
        init_world_point = common.CPoint(corresponding_reference_point.x_ + offset, corresponding_reference_point.y_, corresponding_reference_point.theta_, 1.0/(radius+offset))
        print("init_world_point: ", init_world_point.x_, ", ", init_world_point.y_, ", ", init_world_point.theta_, ", ", init_world_point.curvature_)

        # 第三步,进行world到frenet的坐标转化
        init_frenet_point = transPointToFrenet(init_world_point, global_path, corresponding_index)
        print("init_frenet_point: ", init_frenet_point.x_, ", ", init_frenet_point.y_, ", ", init_frenet_point.theta_, ", ", init_frenet_point.curvature_)

        # 再转回到world系下
        world_point_again = transPointToWorld(init_frenet_point, global_path, corresponding_index)
        print("world_point_again: ", world_point_again.x_, ", ", world_point_again.y_, ", ", world_point_again.theta_, ", ", world_point_again.curvature_)

        # 第四步进行可视化
        # world系下的可视化
        plt.figure()
        plt.axis('equal')
        # 可视化全局导航路径
        plt.plot(point_x, point_y, ':')
        plt.arrow(corresponding_reference_point.x_, corresponding_reference_point.y_, radius * np.cos(corresponding_reference_point.theta_), radius * np.sin(corresponding_reference_point.theta_), fc='r', ec='k', head_width=0.5, head_length=0.5)
        # 可视化点
        plt.arrow(init_world_point.x_, init_world_point.y_, np.cos(init_world_point.theta_), np.sin(init_world_point.theta_), fc='b', ec='k', head_width=0.5, head_length=0.5)
        plt.show()


if __name__ == "__main__":
    test2()