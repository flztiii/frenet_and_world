#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

本代码用于比较汉阳大学规划和本规划之间的差异
author: flztiii

"""

import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from math import *
import tools.common as common
import global_path.cubic_spline as cubic_spline
import path_planning.path_planning_in_frenet as path_planning_in_frenet
import path_planning.path_planning_hanyang as path_planning_hanyang

# 全局变量
VEHICLE_WIDTH = 1.8  # 车辆的宽度[m]
VEHICLE_LENGTH = 4.5  # 车辆的长度[m]
VELOCITY = 5.0  # 车辆行驶速度[m/s]
LOCAL_PLANNING_UPDATE_FREQUENCY = 10.0  # 局部规划更新频率[Hz]
ANIMATE_ON = True  # 是否播放动画
AREA = 25.0  # 动画窗口大小
DISTANCE_TO_GOAL_THRESHOLD = 0.1  # 判断到达终点的距离阈值
OBSTACLE_COST_WEIGHT = 1.0  # 路径选择中障碍物损失的权重
SMOOTH_COST_WEIGHT = 1.0  # 路径选择中平滑损失的权重
CONSISTENCY_COST_WEIGHT = 0.0  # 路径选则中一致性损失的权重
LATERAL_SAMPLING_GAP = 0.2  # 横向采样间隔
LATERAL_SAMPLING_NUM = 20  # 横向采样总数量

# 路径选选择器
class PathSelector:
    def __init__(self):
        self.obstacle_cost_weight_ = OBSTACLE_COST_WEIGHT
        self.smooth_cost_weight_ = SMOOTH_COST_WEIGHT
        self.consistency_cost_weight_ = CONSISTENCY_COST_WEIGHT

    # 进行路径选择
    def select(self, path_candidates, obstacles):
        costs = []
        # 首先判断每一条待选路径的碰撞点
        collision_results, collision_indexs = self.collisionCheck(path_candidates, obstacles)
        # 判断没有碰撞的路径数量
        no_collision_path_num = collision_results.count(0)

        if no_collision_path_num == 0:
            # 全部路径发生碰撞
            # 最优路径就是最长路径
            max_length_index = np.argmax(collision_indexs)
            return max_length_index
        else:
            # 存在无碰撞的路径
            # 遍历每一条路径
            for i, path_candidate in enumerate(path_candidates):
                if collision_results[i] == 1:
                    # 此路径发生碰撞
                    costs.append(float('inf'))
                else:
                    # 此路径没有发生碰撞
                    # 首先计算障碍物损失
                    obstacle_cost = self.calcObstacleCost(i, collision_results)
                    # 其次计算平滑度损失
                    smooth_cost = self.calcSmoothCost(path_candidate)
                    # 最后是一致性损失,不想进行计算
                    consistency_cost = 0.0
                    # 进行加权求和
                    cost = self.obstacle_cost_weight_ * obstacle_cost + self.smooth_cost_weight_ * smooth_cost + self.consistency_cost_weight_ * consistency_cost
                    costs.append(cost)
            # 计算损失最小的路径
            min_cost_index = np.argmin(costs)
            return min_cost_index
    
    # 碰撞检测
    def collisionCheck(self, path_candidates, obstacles):
        collision_index_recorder = []
        collision_results = []
        # 遍历每一条路径
        for _, path in enumerate(path_candidates):
            collision_index = -1
            # 遍历每一个点
            for index, point in enumerate(path.path_):
                # 遍历每一个障碍物
                is_collision = False
                for obstacle in obstacles:
                    # 判断车辆是否与障碍物重叠
                    x, y = common.coordinateTransform(obstacle[0], obstacle[1], point)
                    if x < VEHICLE_LENGTH * 0.5 and x > - VEHICLE_LENGTH * 0.5 and y < VEHICLE_WIDTH * 0.5 and y > - VEHICLE_WIDTH * 0.5:
                        # 障碍物处于车辆的矩形框内,发生碰撞
                        is_collision = True
                        break
                # 判断是否发生碰撞
                if is_collision:
                    # 发生碰撞,记录碰撞点
                    collision_index = index
                    break
            # 判断是否发生碰撞
            if collision_index != -1:
                # 发生碰撞
                collision_results.append(1)
            else:
                # 没有发生碰撞
                collision_results.append(0)
            # 保存碰撞点
            collision_index_recorder.append(collision_index)
        return collision_results, collision_index_recorder


    # 计算障碍物损失
    def calcObstacleCost(self, index, collision_results):
        # 首先确定窗口大小N
        N = len(collision_results)
        # 之后确认高斯参数
        sigma = LATERAL_SAMPLING_GAP * 2
        collision_risk = 0.0
        for i in range(0, len(collision_results)):
            collision_risk += float(collision_results[i]) * common.gaussian(float(index) * LATERAL_SAMPLING_GAP,float(i) * LATERAL_SAMPLING_GAP, sigma)
        cost = collision_risk
        return cost

    # 计算平滑度损失
    def calcSmoothCost(self, path):
        cost = 0.0
        for point in path.path_:
            cost += point.curvature_ ** 2
        return cost


# 计算期望的规划距离
def expectPlanningDistance(velocity):
    return 0.8 * velocity + 4.0

# 进行局部规划过程
def PlanningProcess(global_spline, init_point, goal_point, local_path_planner, obstacles):
    # 验证输入的正确性
    assert(isinstance(global_spline, common.Navigation) and isinstance(init_point, common.CPoint) and isinstance(goal_point, common.CPoint))
    # 定义当前位置
    current_pose = init_point
    # 记录行驶规划
    traveling_recorder = []
    path_candidates_recorder = []
    planned_path_recorder = []
    # 构建路径选则器
    path_selector = PathSelector()
    # 开始进行规划
    while True:
        # 计算局部规划期望距离
        longitude_offset = expectPlanningDistance(VELOCITY)
        # 生成路径组
        local_path_set = []
        for i in range(0, LATERAL_SAMPLING_NUM):
            lateral_offset = float(i  - LATERAL_SAMPLING_NUM / 2) * LATERAL_SAMPLING_GAP
            local_path = local_path_planner.generateLocalPath(global_spline, current_pose, longitude_offset, lateral_offset)
            local_path_set.append(local_path)
        # 记录生成的路径组
        path_candidates_recorder.append(local_path_set)
        # 选出最优路径
        local_path_index = path_selector.select(local_path_set, obstacles)
        local_path = local_path_set[local_path_index]
        # 记录规划的局部路径
        planned_path_recorder.append(local_path)
        # 计算局部规划路径终点与全局导航终点的距离
        min_distance_to_goal = float('inf')
        for path in local_path_set:
            distance_to_goal = np.sqrt((path.path_[-1].x_ - goal_point.x_) ** 2 + (path.path_[-1].y_ - goal_point.y_) ** 2)
            if distance_to_goal < min_distance_to_goal:
                min_distance_to_goal = distance_to_goal
        if min_distance_to_goal < DISTANCE_TO_GOAL_THRESHOLD:
            # 判断此时局部规划已经到达终点
            traveling_recorder.append(local_path.path_)
            break
        else:
            # 判断此时局部规划没有到达终点
            # 判断到下一次局部重规划,车辆走过的距离
            each_episode_travel_distance = VELOCITY / LOCAL_PLANNING_UPDATE_FREQUENCY
            for i, cpoint in enumerate(local_path.path_):
                if local_path.points_dis_[i] >= each_episode_travel_distance:
                    traveling_recorder.append(local_path.path_[:i])
                    current_pose = local_path.path_[i]
                    break

    return traveling_recorder, planned_path_recorder, path_candidates_recorder

# 显示规划过程动画
def show_animate(global_path, obstacles, travel_recorder, planning_recorder, path_candidates_recorder, title):
    # 验证输入正确性
    assert(len(travel_recorder) == len(planning_recorder) and len(travel_recorder) == len(path_candidates_recorder))
    # 开始显示动画
    for i in range(0, len(travel_recorder)):
        for j in range(0, len(travel_recorder[i])):
            plt.cla()
            plt.axis('equal')
            # 可视化全局导航
            plt.plot(global_path.points_x_, global_path.points_y_, ":")
            # 可视化障碍物点
            plt.plot(obstacles.transpose()[0], obstacles.transpose()[1], "xk")
            # 可视化待选路径组
            for path in path_candidates_recorder[i]:
                plt.plot(path.points_x_, path.points_y_, "b")
            # 可视化局部路径
            plt.plot(planning_recorder[i].points_x_, planning_recorder[i].points_y_, "-r")
            # 可视化当前位置
            plt.arrow(travel_recorder[i][j].x_, travel_recorder[i][j].y_, 0.1 * np.cos(travel_recorder[i][j].theta_), 0.1 * np.sin(travel_recorder[i][j].theta_), fc='b', ec='k', head_width=0.5, head_length=0.5)
            # 可视化窗口
            plt.xlim(travel_recorder[i][j].x_ - AREA, travel_recorder[i][j].x_ + AREA)
            plt.ylim(travel_recorder[i][j].y_ - AREA, travel_recorder[i][j].y_ + AREA)
            plt.title(title)
            plt.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
            plt.pause(0.0001)
    plt.close()

# 将行驶路径进行格式化
def traveledPathFormat(traveled_path_recorder):
    points_x, points_y, points_yaw, points_curvature = [], [], [], []
    for i in range(0, len(traveled_path_recorder)):
        for j in range(0, len(traveled_path_recorder[i])):
            points_x.append(traveled_path_recorder[i][j].x_)
            points_y.append(traveled_path_recorder[i][j].y_)
            points_yaw.append(traveled_path_recorder[i][j].theta_)
            points_curvature.append(traveled_path_recorder[i][j].curvature_)
    path = common.CPath(points_x, points_y, points_yaw, points_curvature)
    return path

# 测试函数,沿全局导航从起点行驶到终点
def test():
    # 首先给出全局导航路点
    waypoints_x = [0.0, 20.0, 0.0]
    waypoints_y = [0.0, 20.0, 40.0]
    # 构建2d三次样条曲线
    global_spline = cubic_spline.CubicSpline2D(waypoints_x, waypoints_y)
    # 采样间隔
    gap = 0.1
    # 对2d三次样条曲线进行采样
    sample_s = np.arange(0.0, global_spline.s_[-1], gap)
    point_x, point_y = global_spline.calcPosition(sample_s)
    point_yaw = global_spline.calcYaw(sample_s)
    point_kappa = global_spline.calcKappa(sample_s)
    # 构建全局导航路径
    global_path = common.CPath(point_x, point_y, point_yaw, point_kappa)

    # 给出障碍物列表
    obstacles = np.array([[7.7, 4.0], [10.0, 5.6], [16.7, 15.0], [18.3, 18.1], [13.5, 32.2], [10.2, 34.2]])

    # 给出起始位置和目标点
    init_point = global_spline.calcCPoint(global_spline.s_[0])
    goal_point = global_spline.calcCPoint(global_spline.s_[-1] - common.EPS)

    # 进行局部规划
    frenet_local_path_planner = path_planning_in_frenet.localPathPlanningFactory()
    frenet_planning_traveled_path, frenet_planned_path_recorder, frenet_path_candidates_recorder = PlanningProcess(global_spline, init_point, goal_point, frenet_local_path_planner, obstacles)

    # 判断是否显示动画
    if ANIMATE_ON:
        # 显示动画
        show_animate(global_path, obstacles, frenet_planning_traveled_path, frenet_planned_path_recorder, frenet_path_candidates_recorder, "Frenet Planning")

    # 利用汉阳大学2012年规划方法进行局部规划
    hanyang_local_path_planner = path_planning_hanyang.localPathPlanningFactory()
    hanyang_planning_traveled_path, hanyang_planned_path_recorder, hanyang_path_candidates_recorder = PlanningProcess(global_spline, init_point, goal_point, hanyang_local_path_planner, obstacles)

    # 判断是否显示动画
    if ANIMATE_ON:
        # 显示动画
        show_animate(global_path, obstacles, hanyang_planning_traveled_path, hanyang_planned_path_recorder, hanyang_path_candidates_recorder, "HanYang Planning")

    # 进行可视化准备
    traveled_path_1 = traveledPathFormat(frenet_planning_traveled_path)
    traveled_path_2 = traveledPathFormat(hanyang_planning_traveled_path)

    # 进行可视化
    # 可视化行驶路径
    fig_0 = plt.figure()
    fig_0_ax = fig_0.add_subplot(1, 1, 1)
    fig_0_ax.axis('equal')
    # 可视化全局导航路径
    global_path_vis, = fig_0_ax.plot(point_x, point_y, ':')
    # 可视化局部路径(frenet)
    traveled_path_1_vis, = fig_0_ax.plot(traveled_path_1.points_x_, traveled_path_1.points_y_)
    # 可视化局部路径(hanyang)
    traveled_path_2_vis, = fig_0_ax.plot(traveled_path_2.points_x_, traveled_path_2.points_y_)
    # 可视化障碍物点
    obstacles_vis, = fig_0_ax.plot(obstacles.transpose()[0], obstacles.transpose()[1], "xk")
    # 添加网格
    fig_0_ax.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
    # 添加label
    fig_0_ax.set_xlabel('position[m]')
    fig_0_ax.set_ylabel('position[m]')
    # 添加标注
    fig_0_ax.legend([global_path_vis, traveled_path_1_vis, traveled_path_2_vis, obstacles_vis], ['global path', 'traveled path with method 1', 'traveled path with method 2', 'obstacles'], loc='upper right')

    # 可视化朝向随里程的变化
    fig_1 = plt.figure()
    fig_1_ax = fig_1.add_subplot(1, 1, 1)
    # 可视化traveled_path_1的朝向随路程的变化曲线
    traveled_path_1_yaw_vis, = fig_1_ax.plot(traveled_path_1.points_dis_, traveled_path_1.points_yaw_, 'r')
    # 可视化traveled_path_2的朝向随路程的变化曲线
    traveled_path_2_yaw_vis, = fig_1_ax.plot(traveled_path_2.points_dis_, traveled_path_2.points_yaw_, 'b')
    # 添加标注
    fig_1_ax.legend([traveled_path_1_yaw_vis, traveled_path_2_yaw_vis], ['traveled path 1 yaw', 'traveled path 2 yaw'], loc='upper right')
    # 添加label
    fig_1_ax.set_xlabel('distance[m]')
    fig_1_ax.set_ylabel('yaw[rad]')
    # 添加标题
    fig_1_ax.set_title('yaw profile over distance')

    # 可视化曲率随里程的变化曲线
    fig_2 = plt.figure()
    fig_2_ax = fig_2.add_subplot(1, 1, 1)
    # 可视化traveled_path_1的曲率随路程的变化曲线
    traveled_path_1_cur_vis, = fig_2_ax.plot(traveled_path_1.points_dis_, traveled_path_1.points_curvature_, 'r')
    # 可视化traveled_path_2的曲率随路程的变化曲线
    traveled_path_2_cur_vis, = fig_2_ax.plot(traveled_path_2.points_dis_, traveled_path_2.points_curvature_, 'b')
    # 添加标注
    fig_2_ax.legend([traveled_path_1_cur_vis, traveled_path_2_cur_vis], ['traveled path 1 curvature', 'traveled path 2 curvature'], loc='upper right')
    # 添加label
    fig_2_ax.set_xlabel('distance[m]')
    fig_2_ax.set_ylabel('curvature[rad/m]')
    # 添加标题
    fig_2_ax.set_title('curvature profile over distance')

    plt.show()

if __name__ == "__main__":
    test()
