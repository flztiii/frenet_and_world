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
def transPointToWorld(point, velocity, acceleration, global_path, reference_index):
    # 首先得到frenet坐标参数
    l = point.x_
    r = point.y_
    vl = velocity * np.cos(point.theta_)
    vr = velocity * np.sin(point.theta_)
    al = - velocity ** 2 * point.curvature_ * np.sin(point.theta_) + acceleration * np.cos(point.theta_)
    ar = velocity ** 2 * point.curvature_ * np.cos(point.theta_) + acceleration * np.sin(point.theta_)
    print("l->", l, "r->", r, "vl->", vl, "vr->", vr, "al->", al, "ar->", ar)

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
    print("t ", t)
    dX = np.cos(t)
    dY = np.sin(t)
    ddX = - k * np.sin(t)
    ddY = k * np.cos(t)
    dddX = -np.power(k, 2) * np.cos(t) - m * np.sin(t)
    dddY = -np.power(k, 2) * np.sin(t) + m * np.cos(t)

    print("X->", X, "Y->", Y, "dX->", dX, "dY->", dY, "ddX->", ddX, "ddY->", ddY, "dddX->", dddX, "dddY->", dddY)

    # 计算world系坐标参数
    offset = 0
    # offset = offset - offset * r * k + 1.0/3.0 * (- offset ** 3 + 3.0 * offset * r ** 2) * k ** 2 + (offset ** 3 * r - offset * r ** 3) * k ** 3
    # print("offset is ", offset)
    x = X + offset * dX - r * dY
    y = Y + offset * dY + r * dX
    vx = dX*vl + ddX*offset*vl - ddY*r*vl - dY*vr
    vy = dY*vl + ddY*offset*vl + ddX*r*vl + dX*vr
    ax = al*dX - ar*dY - 2*ddY*vl*vr + ddX*np.power(vl,2) + offset*(al*ddX + dddX*np.power(vl,2)) - r*(al*ddY + dddY*np.power(vl,2))
    ay = ar*dX + al*dY + 2*ddX*vl*vr + ddY*np.power(vl,2) + r*(al*ddX + dddX*np.power(vl,2)) + offset*(al*ddY + dddY*np.power(vl,2))
    # 得到参数之后,计算frenet坐标
    theta = common.pi_2_pi(np.arctan2(vy, vx))
    curvature = (ay * vx - ax * vy) / np.power((vx ** 2 + vy ** 2), (3 / 2))


    curvature_compare = ((dX * vl - ddY * r * vl - dY * vr) * (ar * dX + al * dY + ddY * vl**2 + r * (al * ddX + dddX * vl**2) + 2 * ddX * vl * vr) - (dY * vl + ddX * r * vl + dX * vr) * (al * dX - ar * dY + ddX * vl**2 - r * (al * ddY + dddY * vl**2) - 2 * ddY * vl * vr))/((dY * vl + ddX * r * vl + dX * vr)**2 + (dX * vl - ddY * r * vl - dY * vr)**2)**(3/2)

    print("vx->", vx, "vy->", vy, "ax->", ax, "ay->", ay)
    print("cuvature is ", curvature, ", compare is ", curvature_compare)

    world_point = common.CPoint(x, y, theta, curvature)
    return world_point

# 测试函数
def test():
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
    init_frenet_point = common.CPoint(global_path.points_dis_[corresponding_index], 10.0, 0.0, 0.0)
    velocity = 10.0
    acceleration = 0.0
    # 进行frenet到world的坐标转化
    init_world_point = transPointToWorld(init_frenet_point, velocity, acceleration, global_path, corresponding_index)
    print("init_world_point: ", init_world_point.x_, ", ", init_world_point.y_, ", ", init_world_point.theta_, ", ", init_world_point.curvature_)

    # 进行可视化
    plt.figure()
    plt.axis('equal')
    # 可视化全局导航路径
    plt.plot(point_x, point_y, ':')
    # 可视化点
    plt.plot([init_world_point.x_], [init_world_point.y_], 'or')
    plt.show()

if __name__ == "__main__":
    test()