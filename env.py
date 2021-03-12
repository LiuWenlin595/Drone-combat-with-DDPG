from __future__ import absolute_import

import numpy as np
from random import randint, random
import pygame
import math

from objects import *


class World:
    def __init__(self, world_params=None, vis=None):
        # world_params: height, width, carrier, ship_x, drone_x
        self.world_params = world_params
        self.height = world_params['height']
        self.width = world_params['width']
        self.screen_w = 0
        self.screen_h = 0
        # add
        self.drone_Xbound = world_params['drone_Xbound']  # 导弹初始化位置的随机波动范围
        self.drone_Ybound = world_params['drone_Ybound']  # 导弹初始化位置的随机波动范围
        self.enemy_Xbound = world_params['enemy_Xbound']  # 导弹初始化位置的随机波动范围
        self.enemy_Ybound = world_params['enemy_Ybound']  # 导弹初始化位置的随机波动范围

        self.world_map = np.zeros((self.height, self.width))
        self.enemies = dict()
        self.drones = dict()
        # self.reward = 0
        self.vis = vis
        if self.vis:    # 是否可视化
            pygame.init()
            self.clock = pygame.time.Clock()
            self.screen_size_world = None
            self.screen = None
            self.default_resolution = 500   # 高度的分辨率
            self.screen_scale = 1
            self.render_resolution = self.default_resolution
            self.view_size = self.height    # 可视化范围
            # render中的设置项
            self.rendering_surface = None
            self.font = None

    # 友方无人机移动的函数
    def _move_drones(self, actions):
        # update position
        for (drone_name, drone), shift in zip(self.drones.items(), actions):
            drone.step(shift)

    # 敌方无人机随机运动
    def _random_move_enemies(self):
        for enemy_name, enemy in self.enemies.items():
            new_dx = enemy.dx + np.random.uniform(-1, 1) * self.enemy_Xbound
            new_dy = enemy.dy + np.random.uniform(-1, 1) * self.enemy_Ybound
            if 50 <= new_dx <= self.width:
                enemy.dx = new_dx
            if 0 <= new_dy <= self.height:
                enemy.dy = new_dy

    # 敌方无人机打台球的方式运动
    def _reflect_move_enemies(self):
        for enemy_name, enemy in self.enemies.items():
            new_dx = enemy.dx + enemy.vx * np.cos(enemy.angle)
            new_dy = enemy.dy - enemy.vx * np.sin(enemy.angle)  # y需要减小才表示朝上走
            # while循环防止角落情况可能出现多次越界
            while new_dx < 0 or new_dx > self.width or new_dy < 0 or new_dy > self.height:
                if new_dx < 0 or new_dx > self.width:
                    if enemy.angle > 0:  # 从下往上越界,angle角度为正
                        enemy.angle = math.pi - enemy.angle
                    else:   # 从上往下越界,angle角度为负
                        enemy.angle = -math.pi - enemy.angle
                if new_dy < 0 or new_dy > self.height:
                    enemy.angle *= -1   # 水平方向上的反射,直接取负号
                # 以新的angle重新计算一遍新的位置
                new_dx = enemy.dx + enemy.vx * np.cos(enemy.angle)
                new_dy = enemy.dy - enemy.vx * np.sin(enemy.angle)
            enemy.dx = new_dx
            enemy.dy = new_dy
            # print(enemy.dx,enemy.dy)

    # 计算距离的函数
    def _get_distance(self, obj_a, obj_b):
        a = np.asarray([obj_a.dx, obj_a.dy])
        b = np.asarray([obj_b.dx, obj_b.dy])
        dist = np.linalg.norm(a - b)   # 默认二范数，即sqrt (x1-x2)^2+(y1-y2)^2
        return dist

    # 判断导弹群是否被摧毁
    def _is_drones_destroyed(self):
        done = True
        # for drone_name in self.drones.keys():
        for drone_names, drone in self.drones.items():
            # 如果有一个导弹存活，返回false
            if drone.survive:
                return False
        return done

    # 更新地图
    def _update_map(self):
        self.world_map[:] = 20   # 给world的背景上色

        # self._draw_object(self.ufo, 40)
        # self._draw_object(self. player, 60)
        for enemy_name, enemy in self.enemies.items():
            self._draw_object(enemy, 10)

        for drone_name, drone in self.drones.items():
            self._draw_object(drone, 30)

    # 重新发射导弹, 只是初始化导弹但不会初始化敌方无人机
    def resend(self):
        self.drones = dict()
        self._init_drones()
        for enemy_name, enemy in self.enemies.items():
            self._draw_object(enemy, 10)

        for drone_name, drone in self.drones.items():
            self._draw_object(drone, 30)
        state = self._collect_data()   # 收集所有信息的一个函数，用于记录状态
        return state

    # 重新初始化环境
    def reset(self):
        self._init_world()
        # return copy
        state = self._collect_data()
        return state

    # 初始化世界
    def _init_world(self):
        self.world_map = np.zeros((self.height, self.width))

        self.world_map[:] = 20
        self.drones = dict()
        self.enemies = dict()

        self._init_drones()
        self._init_enemies()

        for enemy_name, enemy in self.enemies.items():
            self._draw_object(enemy, 10)

        for drone_name, drone in self.drones.items():
            self._draw_object(drone, 30)

    # 在地图上画出图像,这里用到了i和j(默认都是0),以dx和dy作为左上角做方块
    def _draw_object(self, obj, color, radius=1):
        if obj.survive:
            for i in range(0, radius):
                for j in range(0, radius):
                    # 将地图上object所在的像素点标记出颜色
                    if 0 < int(i + obj.dx) < 100 and 0 < int(j + obj.dy) < 100:
                        self.world_map[int(i + obj.dx)][int(j + obj.dy)] = color
                    obj.i = int(i + obj.dx)
                    obj.j = int(j + obj.dy)

    # 初始化敌方无人机, 一定出生在右边位置
    def _init_enemies(self):
        for i in range(10):
            try:
                enemy_info = self.world_params['enemy_{}'.format(i)]
                enemy = Enemy(params=enemy_info)
                enemy.dx = self.width/2 + np.random.uniform() * self.width/2
                enemy.dy = np.random.uniform() * self.height
                self.enemies[enemy_info['name']] = enemy
            except KeyError:
                continue

    # 初始化友方无人机, 一定出生在左边位置
    def _init_drones(self):
        for i in range(10):
            try:
                drone_info = self.world_params['drone_{}'.format(i)]
                drone = Drone(params=drone_info)
                drone.dx = np.random.uniform() * self.width/2
                drone.dy = np.random.uniform() * self.height
                self.drones[drone_info['name']] = drone
            except KeyError:
                continue

    # 图像引擎相关函数
    @staticmethod
    # 画地块
    def draw_tile(surface, pos, color, scale):
        rect = pygame.Rect(pos[0] * scale, pos[1] * scale, scale, scale)
        pygame.draw.rect(surface, color, rect)

    @staticmethod
    # 画圆
    def draw_circle(surface, color, pos, scale, radius, name):
        # rect = pygame.Circle(pos[0] * scale, pos[1] * scale, scale, scale)
        # Surface, color, pos, radius, width=0
        # print(pos)
        pos = (pos[0] * scale, pos[1] * scale)
        pygame.draw.circle(surface, color, pos, radius)
        if name:
            myfont = pygame.font.SysFont("monospace", 15)
            label = myfont.render(name, 1, (0, 0, 0))
            surface.blit(label, pos)

    # 画边框
    def _render_layout(self, surface):
        col = (23, 131, 215)
        w, h = self.screen_w, self.screen_h
        pygame.draw.line(surface, col, (w, 0), (w, h), 2)

    # 画世界
    def _render_game_world(self, surface, screen_w, screen_h, scale):
        # min_i, min_j = 0, 0
        # max_i = min(self.height, min_i + self.view_size)
        # max_j = min(self.width, min_j + self.view_size)
        for i in range(screen_w):
            for j in range(screen_h):
                # 画地块
                rect = pygame.Rect(i, j, scale, scale)
                pygame.draw.rect(surface, (120, 0, 255), rect)

        for enemy_name, enemy in self.enemies.items():
            enemy.draw(self, surface, (0, 0), scale)

        for drone_name, drone in self.drones.items():
            drone.draw(self, surface, (0, 0), scale)

    # 环境的图像引擎,用到了pygame库函数
    def render(self):
        events = pygame.event.get()
        # print(events)
        if self.screen is None:
            # initialize screen if it's needed
            while self.screen_scale * self.view_size < self.render_resolution:
                self.screen_scale += 1
            # self.screen_scale = 5
            self.screen_size_world = self.screen_scale * self.view_size
            ui_size = max(100, self.render_resolution // 3)   # 保证屏幕4:3的比例
            # self.screen_w = self.screen_size_world + ui_size    # 4/3 的screen_size_world, 666
            self.screen_w = self.screen_size_world
            self.screen_h = self.screen_size_world     # 1 的screen_size_world, 500
            self.rendering_surface = pygame.Surface((self.screen_w, self.screen_h))
            self.font = pygame.font.SysFont(None, self.render_resolution // 32) # 设置字体大小
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        # self._render_game_world(self.rendering_surface, self.screen_scale)
        self._render_game_world(self.rendering_surface, self.screen_w, self.screen_h, self.screen_scale)
        self._render_layout(self.rendering_surface)
        self.screen.blit(self.rendering_surface, (0, 0))

        pygame.display.flip()

    # 得到网络输入的状态
    def _collect_data(self):
        drone_state = np.zeros((1, 2))
        state = np.empty((1, 0))    # num_drone*2+num_ship*2
        for i, (drone_name, drone) in enumerate(self.drones.items()):
            if drone.survive:
                drone_state[:, 0] = drone.dx
                drone_state[:, 1] = drone.dy
            state = np.concatenate((state, drone_state), axis=1)

        enemy_state = np.zeros((1, 2))
        for i, (enemy_name, enemy) in enumerate(self.enemies.items()):
            if enemy.survive:
                enemy_state[:, 0] = enemy.dx
                enemy_state[:, 1] = enemy.dy
            state = np.concatenate((state, enemy_state), axis=1)

        return state[0]


class DronesMap(World):
    def __init__(self, world_params=None, vis=None):
        super(DronesMap, self).__init__(world_params, vis)

    def step(self, actions):

        done = False
        ep_r = [0 for _ in range(len(self.drones))]

        # step for drones and enemy
        self._move_drones(actions)
        # self._random_move_enemies()
        self._reflect_move_enemies()

        # 初始化enemy的reward
        for enemy_name, enemy in self.enemies.items():
            enemy.reward = 1.0

        # evaluate state
        for drone_name, drone in self.drones.items():
            reward = 0
            distance = 0
            min_dist = 999
            target = object

            for enemy_name, enemy in self.enemies.items():
                # enemy.reward = 1
                if drone.survive and enemy.survive:
                    dist = self._get_distance(drone, enemy)
                    if dist < min_dist:
                        min_dist = dist
                        distance = (dist - 5/(1+dist))
                        target = enemy
            for enemy_name, enemy in self.enemies.items():
                if enemy == target:
                    # print("{} has been choice by {}".format(enemy_name, drone_name))
                    # print(enemy.reward)
                    enemy.reward = enemy.reward / 2
            # 因为导弹的速度设置为1.5, 因此这里的回报最好在1~3的范围内, 以便下面的距离增量回报能发挥作用
            if min_dist < 5:
                reward += target.reward * 3
            elif min_dist < 10:
                reward += target.reward
            elif min_dist < 20:
                reward -= target.reward * 0.5
            else:
                reward -= target.reward
            # 距离变小,得到小奖励;距离变大,给予小惩罚
            # y = x - 5/(1+x)   老的-新的
            # todo  这里有个问题, 第一步的距离奖励
            reward += drone.distance - distance
            drone.distance = distance
            ep_r[drone.index] = reward  # 加入列表
        # check drones
        # 如果导弹全部被摧毁,任务失败
        if self._is_drones_destroyed():
            done = True

        self._update_map()

        state = self._collect_data()
        return state, ep_r, done
