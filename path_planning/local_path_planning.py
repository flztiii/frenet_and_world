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
        self.debug_ = True

    # 生成局部路径
    def generateLocalPath(self, global_path, init_point, longitude_offset, lateral_offset=0.0, method=1):
        '''
            global_path为全局导航路径
            init_point为局部规划的起点
            longtitude_offset为局部规划路径在frenet坐标系纵向的距离
            lateral_offset为局部规划路径在frenet坐标系横向的距离
        '''
        
        # 首先验证输入是否正确
        assert isinstance(global_path, common.CPath) and isinstance(init_point, common.CPoint)
        # 第一步，将init_point转化到frenet坐标系下
        frenet_init_point, init_point_index = self.__transPointToFrenet(init_point, global_path)
        if self.debug_:
            print("frenet start point is ", frenet_init_point.x_, ", ", frenet_init_point.y_)
            print("nearest global reference is ", init_point_index, ", position is ", global_path.points_x_[init_point_index], ", ", global_path.points_y_[init_point_index], ", distance is ", global_path.points_dis_[init_point_index])
        # 判断输出是否有效
        assert (frenet_init_point is not None) and (init_point_index is not None), 'init point not find'
        # 第二步，计算目标点的frenet坐标系和对应下标
        # 计算目标点
        frenet_goal_point = common.CPoint(frenet_init_point.x_ + longitude_offset, lateral_offset, 0.0, 0.0)
        # 计算下标
        goal_point_index = None
        for i in range(init_point_index, len(global_path.path_) - 1):
            if global_path.points_dis_[i] <= frenet_goal_point.x_ and global_path.points_dis_[i + 1] > frenet_goal_point.x_:
                goal_point_index = i
                break
        assert (goal_point_index is not None), 'goal point not find'
        if self.debug_:
            print("goal point is ", frenet_goal_point.x_, ", ", frenet_goal_point.y_)
            print("goal global reference is ", goal_point_index, ", position is ", global_path.points_x_[goal_point_index], ", ", global_path.points_y_[goal_point_index], ", distance is ", global_path.points_dis_[goal_point_index])
        # 第三步，得到frenet系下的局部规划路径
        frenet_spline = g2_spline.G2Spline(frenet_init_point, frenet_goal_point)
        # 第四步，对路径进行采样，得到采样点列表
        sample_number = goal_point_index - init_point_index + 1
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
        # 第五步，将路径转化到world坐标系下
        # 有两种转化方法
        if method == 1:
            # 第一种是利用公式直接进行转化
            world_path_x, world_path_y, world_path_yaw, world_path_curvature = [], [], [], []
            for i in range(0, len(samples)):
                point_x, point_y = common.coordinateInvTransform(frenet_path_x[i] - global_path.points_dis_[i + init_point_index], frenet_path_y[i], global_path.path_[i + init_point_index])
                point_yaw = common.pi_2_pi(global_path.path_[i + init_point_index].theta_ + frenet_path_yaw[i])
                point_curvature = frenet_path_curvature[i] + cos(frenet_path_yaw[i]) ** 3 / (1.0 / (global_path.path_[i + init_point_index].curvature_ + common.EPS) - frenet_path_y[i])
                world_path_x.append(point_x)
                world_path_y.append(point_y)
                world_path_yaw.append(point_yaw)
                world_path_curvature.append(point_curvature)
            local_path = common.CPath(world_path_x, world_path_y, world_path_yaw, world_path_curvature)
            return local_path
        else:
            # 第二种，先转化坐标，再利用坐标计算曲率
            world_path_x, world_path_y, world_path_yaw, world_path_curvature = [], [], [], []
            for i in range(0, len(samples)):
                point_x, point_y = common.coordinateInvTransform(frenet_path_x[i] - global_path.points_dis_[i + init_point_index], frenet_path_y[i], global_path.path_[i + init_point_index])
                point_yaw = common.pi_2_pi(global_path.path_[i + init_point_index].theta_ + frenet_path_yaw[i])
                world_path_x.append(point_x)
                world_path_y.append(point_y)
            # 朝向的计算方法为atan(dy, dx)
            for i in range(0, len(samples) - 1):
                # 首先计算x的变化量dx
                dx = world_path_x[i + 1] - world_path_x[i]
                # 之后计算y的变化量dy
                dy = world_path_y[i + 1] - world_path_y[i]
                # 计算朝向
                yaw = np.arctan2(dy, dx)
                world_path_yaw.append(yaw)
            world_path_yaw.append(world_path_yaw[-1])
            # 曲率的计算方法通过dyaw/ds
            for i in range(0, len(samples) - 1):
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
    def __transPointToFrenet(self, point, global_path):
        # 找到给出坐标在全局导航路径的对应下标
        index = common.getIndexofPath(point.x_, point.y_, global_path.path_)
        # 判断计算得到的下标是否有效
        if index is None:
            return None, None
        # 如果有效，将坐标进行转化
        # 首先计算偏移量
        frenet_longitude_position_offset, frenet_laterl_position_offset = common.coordinateTransform(point.x_, point.y_, global_path.path_[index])
        # 再根据偏移量计算坐标
        frenet_longitude_position = frenet_longitude_position_offset + global_path.points_dis_[index]
        frenet_laterl_position = frenet_laterl_position_offset
        # 完成坐标转化后，进行角度和曲率的转化
        # 角度的转化直接相减即可
        frenet_theta = common.pi_2_pi(point.theta_ - global_path.path_[index].theta_)
        # 曲率的转化利用公式(需要进行验证 TOFIX)
        frenet_curvature = point.curvature_ - cos(frenet_theta) ** 3 / (1.0 / (global_path.path_[index].curvature_ + common.EPS) - frenet_laterl_position)
        # 生成新的点
        frenet_point = common.CPoint(frenet_longitude_position, frenet_laterl_position, frenet_theta, frenet_curvature)

        return frenet_point, index

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
    init_point = common.CPoint(20.0, 20.0, 1.5, 0.0)
    # 生成局部路径
    local_path_factory = localPathPlanningFactory()
    # 第一类方法生成曲率
    local_path_1 = local_path_factory.generateLocalPath(global_path, init_point, 40.0)
    # 第二类方法生成曲率
    local_path_2 = local_path_factory.generateLocalPath(global_path, init_point, 40.0, method=2)
    
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
    test()