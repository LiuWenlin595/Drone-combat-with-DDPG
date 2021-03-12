# 无人机追击任务

## 流程介绍：
在可视化战场上，左半部分有三个友方无人机在随机位置生成，右半部分有两个敌方无人机在随机位置生成。

敌方无人机通过函数控制来采取固定轨迹的移动，友方无人机通过DDPG输出的action进行移动。

任务目标是使得友方无人机可以察觉到敌方无人机的位置，并协同合作不断跟踪两个敌方无人机。

### 文件：
env.py：无人机追击任务的环境信息

objects.py：无人机对象类的定义

DDPG.py：DDPG算法以及执行train和test的main函数

// DQN.py：DQN算法，简单实现了算法，训练由其他同学完成

// TD3.py：TD3算法，简单实现了算法，训练由其他同学完成

### 文件夹：
configs：配置文件包，里面有敌友无人机的初始化参数

utils：工具包，直接引用就可以

expDDPG：DDPG算法保存的模型

#### env.py函数介绍：
_move_drones：# 友方无人机移动的函数  (根据角度和合速度，计算sin，cos)

_random_move_enemies：# 敌方无人机随机运动

_reflect_move_enemies：# 敌方无人机打台球的方式运动

_get_distance：# 计算距离的函数

_is_drones_destroyed：# 判断导弹群是否被摧毁

_update_map：# 更新地图

resend：# 重新发射导弹

reset：# 重新初始化环境

_init_world：# 初始化世界

_draw_object： # 在地图上画出图像,这里用到了i和j(默认都是0),以dx和dy作为左上角做方块

_init_enemies：# 初始化敌方无人机(随机初始化位置在这里，配置文件里的位置没用)

_init_drones：# 初始化友方无人机(随机初始化位置在这里，配置文件里的位置没用)

draw_tile：# 画地块

draw_circle：# 画圆

_render_layout：# 画边框

_render_game_world：# 画世界

render：# 环境的图像引擎 (重要，决定了render的大小，颜色等)

step：# 强化学习中执行动作进入下一个时间步的函数 (重要，在这里可以设置回报)

_collect_data：# 得到网络输入的状态 (重要，在这里设置输入到网络的状态)

#### 可视化：
pygame的surface和draw

## 训练：
运行DDPG.py，参数mode设为train

## 测试：
运行DDPG.py，参数mode设为test

## 对象：
友方无人机：位置，速度，角度，编号

敌方无人机：位置，速度，奖励，编号

## DDPG：
Critic的输入是状态+动作，输出是打分，更新是最小二乘 TD误差。

Actor的输入是状态，输出是一个动作，确定性动作的更新是最大化打分。

critic_target和actor_target都是在计算next_q的时候被使用。

target net网络采用参数软替换，tau设置为0.005(有点高)。

学习率设定为1e-4(有点大)。

折扣因子设定为0.95，即主要朝前看20步(有点小)，episode长度为300。

replay buffer设定为2e-6，没有采用PER。

动作的探索噪声采用高斯噪声，整个训练过程固定均值为0标准差为0.15(前期可以尝试增大方差提高探索)。

## 强化学习相关：
obs，action，reward都没有做归一化
### observation space：
状态空间3x2+2x2=10，对应三个友方无人机的xy和两个敌方无人机的xy，因为三个agent共用一个网络，所以把每个step的采样分成三份，每次把需要执行动作的agent的位置放在最前面。
### action space：
输出-pi~pi的一个角度，控制agent移动的方向。
### reward：
每个agent单独计算奖励。

1. 时间惩罚，逗留越长，受到的小惩罚越大。(后来取消agent死亡的设定后，这个惩罚应该没用了)

2. 绝对距离奖励，当距离敌方无人机的最小距离大于一定值或小于一定值时，会获得一定惩罚或奖励。
 
3. 相对距离奖励，当友方无人机距离敌方无人机的最短距离更进一步的时候，获得奖励，距离变远的时候，获得惩罚。 通过设计一个距离的势能函数y = dist-5/(1+dist)，可以保证在正区间内单调且变化平稳，即agent距离敌人较远时前半部分用于计算相对距离奖励，agent距离敌人较近时后半部分用于计算相对距离奖励。

4. 针对出现的三追一的问题做了一些处理：以0.5的概率更换两个敌方无人机输入的位置；如果三个无人机追击目标相同的话，绝对距离奖励减少一半。

## 输出样例及对应演示：
https://github.com/LiuWenlin595/Drone-combat-with-DDPG/blob/master/%E8%A7%86%E9%A2%91%E6%BC%94%E7%A4%BA.mp4

