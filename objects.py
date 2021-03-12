from __future__ import absolute_import

import math
import numpy as np

import pygame


class BaseObject:
    def __init__(self, params):
        # params参数内容：x, y, static, vx, vy, reward, redius, name
        self.params = params
        # init position
        self.dx = params['x']
        self.dy = params['y']

        self.speed = 0
        self.survive = True
        self.static = params['static']
        if not self.static:
            self.vx = params['vx']
            self.vy = params['vy']
        else:
            self.vx = 0
            self.vy = 0
        try:
            self.reward = params['reward']
        except KeyError:
            self.reward = 0

        self.type = None
        self.radius = params['radius']
        try:
            self.name = params['name']
        except KeyError:
            self.name = None
        try:
            self.index = params['index']
        except KeyError:
            self.index = -1

        # position on the map
        self.i = None
        self.j = None
        # speed

    @NotImplementedError
    def step(angle):
        return

    def draw(self, game, surface, pos, scale):
        color = self._color()
        game.draw_circle(surface, color, (self.i, self.j), scale, self.radius, self.name)


class Drone(BaseObject):
    def __init__(self, params):
        super(Drone, self).__init__(params)
        self.type = 'drone'
        self.color = 240, 180, 250
        self.distance = 0

    def destroy(self):
        self.survive = False
        self.color = 0, 0, 0
        self.distance = 0

    def step(self, shift):
        if self.survive:
            v = np.sqrt(np.square(self.vx) + np.square(self.vy))  # 合速度

            # 直接当作角度, 原来的shift表示的是偏转
            angle = shift

            self.vx = v * np.cos(angle)
            self.vy = v * np.sin(angle)

            self.dx += self.vx   # d是位置，step一次就是一秒
            self.dy += self.vy

    def _color(self):
        return self.color


class Enemy(BaseObject):
    def __init__(self, params):
        super(Enemy, self).__init__(params)
        self.angle = params['angle']
        self.type = 'ship'
        self.color = 240, 255, 250

    def _color(self):
        return self.color

    def destroy(self):
        self.survive = False
        self.color = 0, 0, 0


class Carrier(BaseObject):
    def __init__(self, params, name=None):
        super(Carrier, self).__init__(params)
        self.type = 'carrie'
        self.name = name

    def _color(self):
        return 100, 255, 100

    def destroy(self):
        self.survive = False
        self.color = 0, 0, 0
