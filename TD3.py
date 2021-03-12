import argparse
from collections import namedtuple
from itertools import count
from Drone_Combat.utils.config import load_config

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from Drone_Combat.env import DronesMap

parser = argparse.ArgumentParser()

# mode = 'train' or 'test'
parser.add_argument('--mode', default='train', type=str)
# target smoothing coefficient
parser.add_argument('--tau', default=0.001, type=float)
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=50, type=int)
parser.add_argument('--config_path', '-c', type=str, default='configs/init_env.yml', metavar='PATH')

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=50000,
                    type=int)  # replay buffer size
# parser.add_argument('--num_iteration', default=200,   # 100000
#                     type=int)  # num of  games
parser.add_argument('--batch_size', default=128, type=int)  # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--render', default=True, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('--load', default=True, type=bool)  # load model
# after render_interval, the env.render() will work
parser.add_argument('--render_interval', default=10, type=int)
parser.add_argument('--policy_noise', default=0.03, type=float)
parser.add_argument('--noise_clip', default=0.1, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.03, type=float)
parser.add_argument('--max_episode', default=50000, type=int)
parser.add_argument('--max_length_of_trajectory', default=500, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()

# device = torch.device('cuda:1') if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# __file__ = G:/IDEA/IntelliJ IDEA 2018.2.4/IdeaProject/ReinforcementLearning/environment_DDPG/TD3.py
script_name = os.path.basename(__file__)
config = load_config(args.config_path)
env = DronesMap(world_params=config['world_params'], vis=args.render)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = 16
action_dim = 3
max_action = 1.0
min_Val = torch.tensor(1e-7).float().to(device)  # min value

directory = './exp' + script_name + '/'


class Replay_buffer():

    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        # 如果内存满了就以ptr为索引加入,如果没满就直接append
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        # cur_state, next_state, action, reward, done
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
        # -1是正整数通配符
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


# 输出取得最大Q值的动作
class Actor(nn.Module):   # 继承torch.nn.module

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, 200)
        self.fc4 = nn.Linear(200, action_dim)
        self.bn2 = nn.BatchNorm1d(action_dim)
        self.max_action = max_action
        for m in self.modules():   # 神经网络层的迭代器
            # 初始化各网络层的参数
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        # 激励函数leakyRelu
        a = F.leaky_relu(self.fc1(state))
        a = F.leaky_relu(self.fc2(a))
        a = self.bn1(a)
        a = F.leaky_relu(self.fc3(a))
        a = torch.tanh(self.bn2(self.fc4(a))) * self.max_action    # -1~1 * max_action
        return a

# 判别器,模拟出动作的真实Q值
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.bn = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, 200)
        self.fc4 = nn.Linear(200, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = F.leaky_relu(self.fc1(state_action))
        q = F.leaky_relu(self.fc2(q))
        q = self.bn(q)
        q = F.leaky_relu(self.fc3(q))
        q = self.fc4(q)
        return q


class TD3:
    def __init__(self, state_dim, action_dim, max_action):

        # twin的思想,critic1和critic2初始化参数不同,导致最后的输出结果不同
        # DQN思想:target用于计算next_state
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), args.learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), args.learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), args.learning_rate)

        # copy perameters
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args.capacity)
        # 模型记录
        self.writer = SummaryWriter(directory)
        # delay的思想
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    # 选择动作
    def select_action(self, state):
        # 将模块设置为评估模式
        self.actor.eval()
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, num_iteration):
        self.actor.train()
        # if self.num_training % 500 == 0:
        #     print("====================================")
        #     print("model has been trained for {} times...".format(self.num_training))
        #     print("====================================")
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # 对动作加入噪声,平滑Q
            # Select next action according to target policy:
            # 生成等尺寸全1矩阵
            noise = torch.ones_like(action).data.normal_(
                0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)   # clamp相当于clip
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            # twin的思想, 选择Q值小的
            # detach()是截断反向传播的梯度流
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            # 两个critic分别进行训练
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()  # 执行单步优化
            # tensorboard操作
            self.writer.add_scalar(
                'Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar(
                'Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if num_iteration % args.policy_delay == 0:    # default=2
                # 每两步更新一次actor, 而critic的更新在if外, 即每一步都更新critic
                # Compute actor loss:
                # actor的训练目的就是使输出action的Q值越大越好, 这里直接拿Q值进行loss
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar(
                    'Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                # 两个iteration进行一次参数软替换
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        ((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(
                        ((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(
                        ((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
        self.num_training += 1

    # 保存
    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.actor_target.state_dict(),
                   directory + 'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory + 'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(),
                   directory + 'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory + 'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(),
                   directory + 'critic_2_target.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(
            torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(
            torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(
            torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    agent = TD3(state_dim, action_dim, max_action)
    ep_r = 0

    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            env.reset()
            state = env.resend()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, count_hit, count_destroy = env.step(action)
                ep_r += reward
                # if args.render and i >= args.render_interval:
                #     env.render()
                #     env.clock.tick(10)
                if done or t >= args.max_length_of_trajectory:
                    print(
                        "Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        # 加载已有模型,先注掉
        # if args.load:
        #     agent.load()
        for i in range(args.max_episode):
            state = env.reset()
            env.resend()
            for t in count():   # 计步器
                action = agent.select_action(state)
                # 加入探索噪声
                action = action + np.random.normal(0, args.exploration_noise, size=action_dim)
                action = action.clip(-max_action, max_action)   # 把action的取值范围限制在-1,1
                next_state, reward, done = env.step(action)
                ep_r += reward
                if args.render and i >= args.render_interval:
                    env.render()
                    env.clock.tick(10)
                agent.memory.push(
                    (state, next_state, action, reward, np.float(done)))

                state = next_state
                if done or t >= args.max_length_of_trajectory:    # default=500
                    # param1是名称,param2是纵轴,param3是横轴,tensorboard将我们所需要的数据保存在文件里面供可视化使用
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    if i % args.print_log == 0:   # default=5
                        print("Ep_i  {}, the ep_r is  {:0.2f}, the step is  {}".format(i, ep_r, t))
                    ep_r = 0
                    break
            if i % 10 == 0:
                print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
            # memory满的时候开始训练
            if len(agent.memory.storage) >= args.capacity - 1:
                agent.update(10)
            if i % args.log_interval == 0:  # default=50
                agent.save()

        agent.writer.close()
    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    main()
