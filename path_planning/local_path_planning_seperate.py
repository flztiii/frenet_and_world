#! /usr/bin/python3
#! -*- coding:utf-8 -*-

"""

本代码用于跟随全局导航路径生成局部规划路径
author: flztiii

"""

import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from math import *
import tools.common as common
import path_planning.g2_spline as g2_spline
import global_path.cubic_spline as cubic_spline

# 给定导航路径，定位信息，生成局部路径
class localPathPlanningFactory:
    def __init__(self):
        self.debug_ = False;
        pass
    
    # 生成局部路径
    def generateLocalPath(self, global_path, init_point, longitude_offset, lateral_offset = 0.0, method=1):
        '''
            global_path为全局导航路径
            init_point为局部规划的起点
        '''
        # 首先验证输入是否正确
        assert isinstance(global_path, common.CPath) and isinstance(init_point, common.CPoint)
        assert (len(global_path.path_) > 1)
        # 第一步，将init_point转化到frenet坐标系下
        frenet_init_point, init_point_index = self.__transPointToFrenet(init_point, global_path)
        # 判断输出是否有效
        assert (frenet_init_point is not None) and (init_point_index is not None), 'init point not find'
        print("frenet start point is ", frenet_init_point.x_, ", ", frenet_init_point.y_)
        print("nearest global reference is ", init_point_index, ", position is ", global_path.points_x_[init_point_index], ", ", global_path.points_y_[init_point_index], ", distance is ", global_path.points_dis_[init_point_index])
        # 如果是debug模式,进行可视化
        if self.debug_:
            print("init point is ", init_point.x_, init_point.y_ ,init_point.theta_, init_point.curvature_)
            print("corresponding reference is ", global_path.points_x_[init_point_index], global_path.points_y_[init_point_index], global_path.points_yaw_[init_point_index], global_path.points_curvature_[init_point_index])
            print("frenet point is", frenet_init_point.x_, frenet_init_point.y_, frenet_init_point.theta_, frenet_init_point.curvature_)
            fig = plt.figure()
            plt.plot(global_path.points_x_, global_path.points_y_)
            plt.plot([global_path.points_x_[init_point_index]], [global_path.points_y_[init_point_index]], 'o')
            plt.plot([init_point.x_], [init_point.y_], 'o')
            plt.show()
        
        # 计算目标点
        frenet_goal_point = common.CPoint(frenet_init_point.x_ + longitude_offset, lateral_offset, 0.0, 0.0)
        # 计算下标
        goal_point_index = None
        for i in range(init_point_index, len(global_path.path_) - 1):
            if global_path.points_dis_[i] <= frenet_goal_point.x_ and global_path.points_dis_[i + 1] > frenet_goal_point.x_:
                goal_point_index = i
                break
        assert (goal_point_index is not None), 'goal point not find'
        print("goal point is ", frenet_goal_point.x_, ", ", frenet_goal_point.y_)
        print("goal global reference is ", goal_point_index, ", position is ", global_path.points_x_[goal_point_index], ", ", global_path.points_y_[goal_point_index], ", distance is ", global_path.points_dis_[goal_point_index])
        # 第三步，得到frenet系下的局部规划路径
        frenet_spline = g2_spline.G2Spline(frenet_init_point, frenet_goal_point)
        # 第四步，对路径进行采样，得到采样点列表
        sample_number = (goal_point_index - init_point_index) * 10 + 1
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
        # 如果是debug模式,进行可视化frenet坐标系下的路径
        if self.debug_:
            # 可视化frenet坐标下的导航路径
            frenet_reference_lateral_offsets = [0.0] * len(global_path.points_dis_)
            fig = plt.figure()
            plt.plot(global_path.points_dis_, frenet_reference_lateral_offsets)
            # 可视化对应的起点和终点
            plt.plot([global_path.points_dis_[init_point_index]], [frenet_reference_lateral_offsets[init_point_index]], 'o')
            plt.plot([global_path.points_dis_[goal_point_index]], [frenet_reference_lateral_offsets[goal_point_index]], 'o')
            # 可视化生成的路径
            plt.plot(frenet_path_x, frenet_path_y)
            plt.show()
        frenet_path = common.CPath(frenet_path_x, frenet_path_y, frenet_path_yaw, frenet_path_curvature)
        print("frenet_path size is ", len(frenet_path.path_), ", samples number is ", len(samples))
        print("global_path size is ", len(global_path.path_))
        # 第五步，将路径转化到world坐标系下
        # 有两种转化方法
        if method == 1:
            # 第一种是利用公式直接进行转化
            last_corresponding_index = 0
            world_path_x, world_path_y, world_path_yaw, world_path_curvature = [], [], [], []
            for i in range(init_point_index, goal_point_index + 1):
                # 找到对应的下标
                corresponding_index = last_corresponding_index + 1
                for index in range(corresponding_index, len(samples)):
                    # print("frenet_path.points_x_", len(frenet_path.points_x_), ", index ", index)
                    assert(frenet_path.points_x_[index] > frenet_path.points_x_[index - 1])
                    distance = frenet_path.points_x_[index] - global_path.points_dis_[i]
                    if distance > 0.0:
                        corresponding_index = index
                        break
                last_corresponding_index = corresponding_index

                world_cpoint = self.__transPointToWorld(frenet_path.path_[corresponding_index], global_path, i)
                world_path_x.append(world_cpoint.x_)
                world_path_y.append(world_cpoint.y_)
                world_path_yaw.append(world_cpoint.theta_)
                world_path_curvature.append(world_cpoint.curvature_)
            local_path = common.CPath(world_path_x, world_path_y, world_path_yaw, world_path_curvature)
            return local_path
        elif method == 2:
            # 第二种，先转化坐标，再利用坐标计算曲率
            world_path_x, world_path_y, world_path_yaw, world_path_curvature = [], [], [], []
            offsets = []
            last_corresponding_index = 0
            for i in range(init_point_index, goal_point_index + 1):
                # point_x, point_y = common.coordinateInvTransform(frenet_path_x[i] - global_path.points_dis_[i + init_point_index], frenet_path_y[i], global_path.path_[i + init_point_index])
                # point_yaw = common.pi_2_pi(global_path.path_[i + init_point_index].theta_ + frenet_path_yaw[i])
                # world_path_x.append(point_x)
                # world_path_y.append(point_y)
                # world_path_yaw.append(point_yaw)
                # 找到对应的下标
                corresponding_index = last_corresponding_index + 1
                for index in range(corresponding_index, len(samples)):
                    # print("frenet_path.points_x_", len(frenet_path.points_x_), ", index ", index)
                    assert(frenet_path.points_x_[index] > frenet_path.points_x_[index - 1])
                    distance = frenet_path.points_x_[index] - global_path.points_dis_[i]
                    if distance > 0.0:
                        corresponding_index = index
                        break
                last_corresponding_index = corresponding_index

                world_cpoint = self.__transPointToWorld(frenet_path.path_[corresponding_index], global_path, i)
                world_path_x.append(world_cpoint.x_)
                world_path_y.append(world_cpoint.y_)
                offsets.append(frenet_path.points_x_[corresponding_index] - global_path.points_dis_[i])
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

    
    # 将坐标转化到frenet系下
    def  __transPointToFrenet(self, point, global_path):
        # 找到给出坐标在全局导航路径的对应下标
        index = common.getIndexofPath(point.x_, point.y_, global_path.path_)
        # 判断计算得到的下标是否有效
        if index is None:
            return None, None
        # 如果有效，将坐标进行转化
        # 首先得到全局导航参考点的参数
        X = global_path.points_x_[index]
        Y = global_path.points_y_[index]
        t = global_path.points_yaw_[index]
        k = global_path.points_curvature_[index]
        m = 0.0
        # 计算m=dk/ds, 曲率的导数
        if index > 0:
            ds = global_path.points_dis_[index] - global_path.points_dis_[index - 1]
            dk = global_path.points_curvature_[index] - global_path.points_curvature_[index - 1]
            assert ds != 0
            m = dk / ds
        else:
            ds = global_path.points_dis_[index + 1] - global_path.points_dis_[index]
            dk = global_path.points_curvature_[index + 1] - global_path.points_curvature_[index]
            assert ds != 0
            m = dk / ds
        dX = np.cos(t)
        dY = np.sin(t)
        ddX = - k * np.sin(t)
        ddY = k * np.cos(t)
        dddX = -np.power(k, 2) * np.cos(t) - m * np.sin(t)
        dddY = -np.power(k, 2) * np.sin(t) + m * np.cos(t)

        # 其次得到车辆初始位置相关参数
        x = point.x_;
        y = point.y_;
        vx = np.cos(point.theta_)
        vy = np.sin(point.theta_)
        ax = - point.curvature_ * np.sin(point.theta_)
        ay = point.curvature_ * np.cos(point.theta_)

        # 计算frenet坐标参数
        l, r = common.coordinateTransform(point.x_, point.y_, global_path.path_[index])
        l += global_path.points_dis_[index]
        offset = l - global_path.points_dis_[index]
        offset = offset - offset * r * k
        # offset = offset - offset * r * k + 1.0/3.0 * (- offset ** 3 + 3.0 * offset * r ** 2) * k ** 2 + (offset ** 3 * r - offset * r ** 3) * k ** 3
        # 首先计算frenet转化后的一阶导数
        vl = (dX*vx + dY*vy)*np.power(ddX*dX*offset + ddY*dY*offset - ddY*dX*r + ddX*dY*r + np.power(dX,2) + np.power(dY,2),-1)
        vr = (-((dY + ddY*offset + ddX*r)*vx) + (dX + ddX*offset - ddY*r)*vy)*np.power(dY*(dY + ddY*offset + ddX*r) + dX*(ddX*offset - ddY*r) + np.power(dX,2),-1)
        al = (ax*dX + ay*dY + vl*(-((ddX*dX + ddY*dY + dddX*dX*offset + dddY*dY*offset - dddY*dX*r + dddX*dY*r)*vl) + 2*(ddY*dX - ddX*dY)*vr))*np.power(dY*(dY + ddY*offset + ddX*r) + dX*(ddX*offset - ddY*r) + np.power(dX,2),-1)
        ar = (-(ax*(dY + ddY*offset + ddX*r)) + ay*(dX + ddX*offset - ddY*r) + vl*(-2*vr*(ddX*dX + ddY*(dY + ddY*offset) + offset*np.power(ddX,2)) + vl*(ddX*dY + dddX*dY*offset - dddX*dX*r + r*np.power(ddX,2) + r*np.power(ddY,2) + ddY*(-dX + dddX*(np.power(offset,2) + np.power(r,2))) - dddY*(dX*offset + dY*r + ddX*(np.power(offset,2) + np.power(r,2))))))*np.power(dY*(dY + ddY*offset + ddX*r) + dX*(ddX*offset - ddY*r) + np.power(dX,2),-1)
        
        # 得到参数之后,计算frenet坐标
        theta = common.pi_2_pi(np.arctan2(vr, vl))
        curvature = (ar * vl - al * vr) / np.power((vl ** 2 + vr ** 2), (3 / 2))
        frenet_point = common.CPoint(l, r, theta, curvature)

        return frenet_point, index
    
    # 将frenet坐标转到world系下
    def __transPointToWorld(self, point, global_path, reference_index):
        # 首先得到frenet坐标参数
        l = point.x_
        r = point.y_
        vl = np.cos(point.theta_)
        vr = np.sin(point.theta_)
        al = - point.curvature_ * np.sin(point.theta_)
        ar = point.curvature_ * np.cos(point.theta_)

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
        dX = np.cos(t)
        dY = np.sin(t)
        ddX = - k * np.sin(t)
        ddY = k * np.cos(t)
        dddX = -np.power(k, 2) * np.cos(t) - m * np.sin(t)
        dddY = -np.power(k, 2) * np.sin(t) + m * np.cos(t)

        # 计算world系坐标参数
        offset = l - global_path.points_dis_[reference_index]
        offset = offset - offset * r * k
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
        world_point = common.CPoint(x, y, theta, curvature)
        return world_point

# 测试函数(直线引导线)
def test1():
    # 首先建立全局导航路径
    # 初始化散点
    x = [0.0, 20.0, 40.0]
    y = [0.0, 0.0, 0.0]
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
    init_point = common.CPoint(10.0, 1.0, 0.0, 0.0)
    # 生成局部路径
    local_path_factory = localPathPlanningFactory()
    # 第一类方法生成曲率
    local_path_1 = local_path_factory.generateLocalPath(global_path, init_point, 20.0)
    local_path_2 = local_path_factory.generateLocalPath(global_path, init_point, 20.0, method=2)

    # 计算两者的曲率最大差
    curvature_deviations = []
    for i,_ in enumerate(local_path_1.path_):
        curvature_deviations.append(abs(local_path_1.points_curvature_[i] - local_path_2.points_curvature_[i]))
    max_curvature_deviation = max(curvature_deviations)
    print("max curvature deviation is ", max_curvature_deviation)

    # 可视化
    fig_1 = plt.figure()
    fig_1_ax = fig_1.add_subplot(1, 1, 1)
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
    # 绘制曲率随路程的变化曲线
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

# 验证曲率转化公式的正确性(圆引导线)
def test2():
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
    # 给定车辆初始位置
    init_point = common.CPoint(0.0, 30.0, np.pi, 1.0/30.0)
    # 生成局部路径
    local_path_factory = localPathPlanningFactory()
    # 第一类方法生成曲率
    local_path_1 = local_path_factory.generateLocalPath(global_path, init_point, 30.0, -10.0)
    # 第二类方法生成曲率
    local_path_2 = local_path_factory.generateLocalPath(global_path, init_point, 30.0, -10.0, method=2)

    # 计算两者的曲率最大差
    curvature_deviations = []
    for i,_ in enumerate(local_path_1.path_):
        curvature_deviations.append(abs(local_path_1.points_curvature_[i] - local_path_2.points_curvature_[i]))
    max_curvature_deviation = max(curvature_deviations)
    print("max curvature deviation is ", max_curvature_deviation)

    # 可视化
    fig_1 = plt.figure()
    fig_1_ax = fig_1.add_subplot(1, 1, 1)
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
    # 绘制曲率随路程的变化曲线
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

# 测试函数(随机引导线)
def test3():
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
    init_point = common.CPoint(20.0, 20.0, 1.5, 0.0)
    # 生成局部路径
    local_path_factory = localPathPlanningFactory()
    # 第一类方法生成曲率
    local_path_1 = local_path_factory.generateLocalPath(global_path, init_point, 40.0)
    # 第二类方法生成曲率
    local_path_2 = local_path_factory.generateLocalPath(global_path, init_point, 40.0, method=2)

    # 计算两者的曲率最大差
    curvature_deviations = []
    for i,_ in enumerate(local_path_1.path_):
        if i < len(local_path_1.path_) - 2:
            curvature_deviations.append(abs(local_path_1.points_curvature_[i] - local_path_2.points_curvature_[i]))
    max_curvature_deviation = max(curvature_deviations)
    print("max curvature deviation is ", max_curvature_deviation, ", index is ", curvature_deviations.index(max_curvature_deviation))

    # 可视化
    fig_1 = plt.figure()
    fig_1_ax = fig_1.add_subplot(1, 1, 1)
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
    # 绘制曲率随路程的变化曲线
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
    test3()
