import argparse
from itertools import count

import os
import numpy as np
from env import DronesMap
from utils.config import load_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import math
import time
import copy

parser = argparse.ArgumentParser()
# mode = 'train' or 'test'
parser.add_argument('--mode', default='test', type=str)
# target smoothing coefficient
parser.add_argument('--tau', default=0.005, type=float)
# parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=50, type=int)
parser.add_argument('--config_path', '-c', type=str, default='configs/init_env.yml', metavar='PATH',
                    help='path to the config')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.95, type=int)  # discounted factor
parser.add_argument('--capacity', default=2e6, type=int)  # replay buffer size
parser.add_argument('--batch_size', default=256, type=int)  # mini batch size
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=2, type=int)
# optional parameters
# parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--render', default=True, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)  #
parser.add_argument('--load', default=True, type=bool)  # load model
# after render_interval, the env.render() will work
parser.add_argument('--render_interval', default=480, type=int)
parser.add_argument('--exploration_noise', default=0.15, type=float)
parser.add_argument('--max_episode', default=50000, type=int)  # num of games
parser.add_argument('--max_length_of_trajectory',    # 轨迹最大长度
                    default=300, type=int)  # num of games
parser.add_argument('--print_log', default=10, type=int)
parser.add_argument('--update_iteration', default=10, type=int)   # iteration次数
args = parser.parse_args()

# device = 'cpu' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
print(torch.cuda.get_device_properties('cuda:0'))
# __file__ = G:/IDEA/IntelliJ IDEA 2018.2.4/IdeaProject/ReinforcementLearning/Drone_Combat/DDPG.py
script_name = os.path.basename(__file__)

config = load_config(args.config_path)
env = DronesMap(world_params=config['world_params'], vis=args.render)


if args.seed:
    # env.seed(random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

dim_state = config['world_params']['num_drone'] * 2 + config['world_params']['num_enemy'] * 2
dim_action = 1  # 每次只输出一个导弹的动作
n_agent = config['world_params']['num_drone']   # 有多少个导弹
maxAction = math.pi

# directory = './exp' + script_name + '/myenv_random_seed' + str(args.random_seed) + '/'
directory = './exp' + script_name + '/saved_model/'


class ReplayBuffer:

    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        # 随机采样进行训练
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x_list, y_list, u_list, r_list, d_list = [], [], [], [], []

        for i in ind:
            x, y, u, r, d = self.storage[i]
            x_list.append(np.array(x, copy=False))
            y_list.append(np.array(y, copy=False))
            u_list.append(np.array(u, copy=False))
            r_list.append(np.array(r, copy=False))
            d_list.append(np.array(d, copy=False))

        # reshape(-1,1) 表示不知道有多少行,但是列数一定是1
        return np.array(x_list), np.array(y_list), np.array(u_list), \
               np.array(r_list).reshape(-1, 1), np.array(d_list).reshape(-1, 1)


# 只输出单个动作
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l0 = nn.Linear(state_dim, 128)
        self.l1 = nn.Linear(128, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 32)
        self.l4 = nn.Linear(32, action_dim)
        # self.l1 = nn.Linear(state_dim, 128)
        # self.l2 = nn.Linear(128, 256)
        # self.l3 = nn.Linear(256, 128)
        # self.l4 = nn.Linear(128, 64)
        # self.l5 = nn.Linear(64, action_dim)
        self.max_action = max_action
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm1d):
                # x' = x-μ / sqrt(σ^2)
                # y = γ * x' + β
                # m.weight对应γ, m.bias对应β (猜)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.l0(x))
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = self.max_action * torch.tanh(self.l4(x))
        return x


class Critic(nn.Module):
    # Critic的输入为s, v'以及r, 不需要a因为a是由actor决定的, 而且a的表现在TD error里反应在了v'中
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l0_state = nn.Linear(state_dim, 32)
        self.l0_actor = nn.Linear(action_dim, 32)
        self.l1 = nn.Linear(64, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 32)
        self.l4 = nn.Linear(32, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, u):
        # x = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
        x_state = self.l0_state(x)
        x_actor = self.l0_actor(u)
        x = torch.cat([x_state, x_actor], 1)
        x = F.leaky_relu(self.l1(x))
        # x = F.leaky_relu(self.l0_state(x) + self.l0_actor(u))
        # x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = self.l4(x)
        # 根据当前状态, 输出Critic认为准确的Q值
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), args.learning_rate, weight_decay=1e-2)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), args.learning_rate, weight_decay=1e-2)

        self.replay_buffer = ReplayBuffer()
        # 仅仅为了在tensorboardX可视化中使用, 用来记录网络的更新次数, 和target参数替换无关
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0

    def select_action(self, state):
        # model.eval()
        # 不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False
        self.actor.eval()
        action = []
        # 对每个无人机采用同一个actor进行动作选取
        for i in range(n_agent):
            sub_state = copy.copy(state)  # 深拷贝
            # 更换当前无人机的位置到中心位置, 其余agent顺次往后移动两位
            tmp_dx, tmp_dy = sub_state[2*i], sub_state[2*i+1]
            for j in range(2*i+1, 1, -1):
                sub_state[j] = sub_state[j-2]
            sub_state[0] = tmp_dx
            sub_state[1] = tmp_dy
            # 转成tensor进行计算
            sub_state = torch.FloatTensor(sub_state.reshape(1, -1)).to(device)
            sub_action = self.actor(sub_state).cpu().data.numpy().flatten()
            action.append(sub_action[0])
        return action

    def update(self):
        # model.train()
        # 启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为True

        self.actor.train()
        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.unsqueeze(torch.FloatTensor(u).to(device), -1)  # 在最后扩展一个维度, 为了匹配网络输入
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(d).to(device)
            # Compute the target Q value
            # q_target = self.r + self.gamma * max_q_next
            next_q = self.critic_target(next_state, self.actor_target(next_state))
            # detach截断next_q的反向传播, 保证了理论上的半梯度更新
            # detach工作原理 :
            # 1. copy() 将原来的tensor复制一份  (并不会影响原来tensor的反向传播)
            # 2. 将其requires_grad属性改为False    (默认为True)
            target_q = reward + ((1 - done) * args.gamma * next_q).detach()

            # Get current Q estimate
            # current_q = self.critic(state, action)
            current_q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(target_q, current_q)
            self.writer.add_scalar(
                'Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)

            # Optimize the critic
            # 将梯度初始化为零
            self.critic_optimizer.zero_grad()
            # 反向传播求梯度
            critic_loss.backward()
            # 更新所有网络参数, Adam梯度下降往最小值方向逼近
            # 经过调试, 确实梯度有更新, 但是不知道具体是怎么更新的
            self.critic_optimizer.step()

            # Compute actor loss
            # todo
            # 这里以后可能需要改, actor是分布式的, 不应该由一个集中式Actor全局控制
            # batchsizex1的大小, 因此需要mean
            # action是在轨迹库中根据采样时actor参数输出的动作, self.actor(state)是根据当前actor参数输出的动作
            # 因此critic评价的并不是当前样本的动作, 而是当前actor输出的动作
            # 但是无论采用那个动作, 都可以对actor进行一次 在当前state采取什么动作能获得更高评价的一个更新
            # 并且这里必须用self.actor(state), 如果用action的话无法根据actor_loss对actor参数进行更新
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar(
                'Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            # 将梯度初始化为零
            self.actor_optimizer.zero_grad()
            # 反向传播求梯度
            actor_loss.backward()
            # 更新所有网络参数, Adam梯度下降往最小值方向逼近
            self.actor_optimizer.step()

            # Update the frozen target models
            # 每个iteration对Critic参数进行软替换
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data)
            self.num_critic_update_iteration += 1
            # 每隔3个iteration对Actor参数进行软替换
            if it % 3 == 0:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data)
            self.num_actor_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth', map_location=device))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth', map_location=device))
        self.actor_target.load_state_dict(torch.load(directory + 'actor.pth', map_location=device))
        self.critic_target.load_state_dict(torch.load(directory + 'critic.pth', map_location=device))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    agent = DDPG(dim_state, dim_action, maxAction)
    ep_r = [0 for _ in range(n_agent)]
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            # state = env.resend()
            for t in count():
                action = agent.select_action(state)
                # print(action)
                next_state, reward, done = env.step(action)
                for k in range(n_agent):
                    ep_r[k] += reward[k]
                if args.render:     # and i >= args.render_interval:
                    env.render()
                    env.clock.tick(10)
                if t % args.print_log == 0:
                    for k in range(n_agent):
                        print("Ep_i:  {}, the ep_r is:  {:0.2f}, the step is:  {}.".format(i, ep_r[k], t))
                if done or t >= args.max_length_of_trajectory:
                    for k in range(n_agent):
                        print("Ep_i:  {}, the ep_r is:  {:0.2f}, the step is:  {}.".format(i, ep_r[k], t))
                    ep_r[:] = [0]*n_agent
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")

        try:
            if args.load:
                agent.load()
            print("load successfully!")
        except:
            print(directory)
            print("load failed")

        for i in range(args.max_episode):
            state = env.reset()
            # resend的作用是单纯的初始化友方无人机而不初始化敌方无人机, 在分批次任务中使用
            # state = env.resend()
            for t in count():

                action = agent.select_action(state)
                # add noise to action
                action = (action + np.random.normal(0, args.exploration_noise, size=n_agent)).clip(
                    -maxAction, maxAction)  # 约束范围在maxAction之间

                next_state, reward, done = env.step(action)     # 这里的reward要拆成三份
                for k in range(n_agent):
                    ep_r[k] += reward[k]
                # # 训练的时候可以看一下
                # if args.render and i >= args.render_interval:
                #     env.render()
                #     env.clock.tick(10)
                for k in range(n_agent):
                    # 更改当前状态
                    sub_state = copy.copy(state)    # 深拷贝
                    tmp_dx, tmp_dy = sub_state[2*k], sub_state[2*k+1]
                    for j in range(2*k+1, 1, -1):
                        sub_state[j] = sub_state[j-2]
                    sub_state[0] = tmp_dx
                    sub_state[1] = tmp_dy
                    # 更改next状态
                    sub_next_state = copy.copy(next_state)    # 深拷贝
                    tmp_dx, tmp_dy = sub_next_state[2*k], sub_next_state[2*k+1]
                    for j in range(2*k+1, 1, -1):
                        sub_next_state[j] = sub_next_state[j-2]
                    sub_next_state[0] = tmp_dx
                    sub_next_state[1] = tmp_dy

                    # # 0.5的概率换位置存储e1和e2
                    if np.random.rand() >= 0.5:
                        state[6], state[7], state[8], state[9] = state[8], state[9], state[6], state[7]
                        next_state[6], next_state[7], next_state[8], next_state[9] = \
                            next_state[8], next_state[9], next_state[6], next_state[7]

                    # 一个轨迹分三次采样
                    agent.replay_buffer.push((sub_state, sub_next_state, action[k], reward[k], done))
                state = next_state

                if done or t >= args.max_length_of_trajectory:
                    for k in range(n_agent):
                        agent.writer.add_scalar('ep_r'+str(k), ep_r[k], global_step=i)
                    if i % args.print_log == 0:
                        for k in range(n_agent):
                            print("Ep_i:  {}, the ep_r is:  {:0.2f}, the step is:  {}.".format(i, ep_r[k], t))
                    ep_r[:] = [0]*n_agent
                    break

            # memory检测, replay buffer达到一定数量的时候开始训练
            if i % 10 == 0:
                print('Episode {},  The memory size is {} '.format(i, len(agent.replay_buffer.storage)))
            if len(agent.replay_buffer.storage) >= 2e4:
                agent.update()
            if i % args.log_interval == 0:
                agent.save()

    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    main()
