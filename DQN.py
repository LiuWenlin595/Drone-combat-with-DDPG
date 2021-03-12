import argparse
from itertools import count

import os
import numpy as np
from Drone_Combat.env import DronesMap
from Drone_Combat.utils.config import load_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
# mode = 'train' or 'test'
parser.add_argument('--mode', default='train', type=str)
# target smoothing coefficient
parser.add_argument('--tau', default=0.001, type=float)
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)
parser.add_argument('--config_path', '-c', type=str, default='configs/init_env.yml', metavar='PATH',
                    help='path to the config')
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=50000,
                    type=int)  # replay buffer size
parser.add_argument('--batch_size', default=128, type=int)  # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--render', default=True, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)  #
parser.add_argument('--load', default=True, type=bool)  # load model
# after render_interval, the env.render() will work
parser.add_argument('--render_interval', default=10, type=int)
parser.add_argument('--exploration_noise', default=0.03, type=float)
parser.add_argument('--max_episode', default=50000, type=int)  # num of games
parser.add_argument('--max_length_of_trajectory',    # 轨迹最大长度
                    default=500, type=int)  # num of games
parser.add_argument('--print_log', default=10, type=int)
parser.add_argument('--update_iteration', default=10, type=int)   # iteration次数

parser.add_argument('--n_actions', default=11, type=int)   # 把shift角度切成几份
args = parser.parse_args()

device = 'cpu' if torch.cuda.is_available() else 'cpu'
# device = torch.device('cuda:1') if torch.cuda.is_available() else 'cpu'
print(device)
# __file__ = G:/IDEA/IntelliJ IDEA 2018.2.4/IdeaProject/ReinforcementLearning/environment_DDPG/DDPG.py
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
min_Valmin_Val = torch.tensor(1e-7).float().to(device)  # min value

directory = './exp' + script_name + '/'

class Replay_buffer:

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
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class Q_Value(nn.Module):
    def __init__(self, state_dim):
        super(Q_Value, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.bn = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 200)
        self.l4 = nn.Linear(200, args.n_actions)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 这里只需要把state输入就可以了
    def forward(self, state):
        q = F.leaky_relu(self.l1(state))
        q = F.leaky_relu(self.l2(q))
        q = self.bn(q)
        q = F.leaky_relu(self.l3(q))
        q = self.l4(q)
        # actions_value1 = self.q_eval1(state).cpu().data.numpy().flatten()
        # action1 = np.argmax(actions_value1)
        # q = q.cpu().data.numpy().flatten()
        # q = np.argmax(q)
        return q

class DQN(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.q_eval1 = Q_Value(state_dim).to(device)
        self.q_eval2 = Q_Value(state_dim).to(device)
        self.q_eval3 = Q_Value(state_dim).to(device)
        self.q_eval4 = Q_Value(state_dim).to(device)
        self.q_eval5 = Q_Value(state_dim).to(device)
        self.q_next1 = Q_Value(state_dim).to(device)
        self.q_next2 = Q_Value(state_dim).to(device)
        self.q_next3 = Q_Value(state_dim).to(device)
        self.q_next4 = Q_Value(state_dim).to(device)
        self.q_next5 = Q_Value(state_dim).to(device)

        self.q_next1.load_state_dict(self.q_eval1.state_dict())
        self.q_next2.load_state_dict(self.q_eval2.state_dict())
        self.q_next3.load_state_dict(self.q_eval3.state_dict())
        self.q_next4.load_state_dict(self.q_eval4.state_dict())
        self.q_next5.load_state_dict(self.q_eval5.state_dict())
        self.q_eval_optimizer1 = optim.Adam(self.q_eval1.parameters(), args.learning_rate)
        self.q_eval_optimizer2 = optim.Adam(self.q_eval2.parameters(), args.learning_rate)
        self.q_eval_optimizer3 = optim.Adam(self.q_eval3.parameters(), args.learning_rate)
        self.q_eval_optimizer4 = optim.Adam(self.q_eval4.parameters(), args.learning_rate)
        self.q_eval_optimizer5 = optim.Adam(self.q_eval5.parameters(), args.learning_rate)

        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)
        self.num_q_update_iteration = 0
        self.num_training = 0

    # 在这里出现了问题
    # DQN要求Q值网络输出所有动作对应的Q值, 然后选择其中Q值最大的动作
    # 可是shift是角度, 网络难以有这么多输出，且难以在这么多输出之中搜索最大的Q值
    # 所以要考虑离散的DQN
    def select_action(self, state):
        self.q_eval1.eval()
        self.q_eval2.eval()
        self.q_eval3.eval()
        self.q_eval4.eval()
        self.q_eval5.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        actions_value1 = self.q_eval1(state).cpu().data.numpy().flatten()
        action1 = np.argmax(actions_value1)
        actions_value2 = self.q_eval2(state).cpu().data.numpy().flatten()
        action2 = np.argmax(actions_value2)
        actions_value3 = self.q_eval3(state).cpu().data.numpy().flatten()
        action3 = np.argmax(actions_value3)
        actions_value4 = self.q_eval4(state).cpu().data.numpy().flatten()
        action4 = np.argmax(actions_value4)
        actions_value5 = self.q_eval5(state).cpu().data.numpy().flatten()
        action5 = np.argmax(actions_value5)
        # shift = max_action - number * 2 * max_action/args.n_actions
        action = np.array([action1, action2, action3, action4, action5])
        return action

    def update(self):
        self.q_eval1.train()
        self.q_eval2.train()
        self.q_eval3.train()
        self.q_eval4.train()
        self.q_eval5.train()
        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            # action = torch.FloatTensor(u).to(device)
            action = torch.from_numpy(u)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)
            # select next_action and Compute the target Q value
            self.q_next1.eval()
            self.q_next2.eval()
            self.q_next3.eval()
            self.q_next4.eval()
            self.q_next5.eval()
            actions_value1 = self.q_next1(next_state).detach().numpy()
            target_Q1 = np.argmax(actions_value1, axis=1)
            actions_value2 = self.q_next2(next_state).detach().numpy()
            target_Q2 = np.argmax(actions_value2, axis=1)
            actions_value3 = self.q_next3(next_state).detach().numpy()
            target_Q3 = np.argmax(actions_value3, axis=1)
            actions_value4 = self.q_next4(next_state).detach().numpy()
            target_Q4 = np.argmax(actions_value4, axis=1)
            actions_value5 = self.q_next5(next_state).detach().numpy()
            target_Q5 = np.argmax(actions_value5, axis=1)
            # shift = max_action - number * 2 * max_action/args.n_actions
            target_Q = ((target_Q1+target_Q2+target_Q3+target_Q4+target_Q5)/5).astype(np.float32)
            target_Q = target_Q[:, np.newaxis]
            target_Q = reward + ((1 - done) * args.gamma * target_Q)

            # Get current Q estimate
            haha = self.q_eval1(state).cpu().data.numpy().flatten()
            current_Q1 = self.q_eval1(state)
            current_Q2 = self.q_eval1(state)
            current_Q3 = self.q_eval1(state)
            current_Q4 = self.q_eval1(state)
            current_Q5 = self.q_eval1(state)
            current_Q1 = torch.gather(current_Q1, dim=1, index=action[:, 0].long().unsqueeze(dim=1))
            current_Q2 = torch.gather(current_Q2, dim=1, index=action[:, 1].long().unsqueeze(dim=1))
            current_Q3 = torch.gather(current_Q3, dim=1, index=action[:, 2].long().unsqueeze(dim=1))
            current_Q4 = torch.gather(current_Q4, dim=1, index=action[:, 3].long().unsqueeze(dim=1))
            current_Q5 = torch.gather(current_Q5, dim=1, index=action[:, 4].long().unsqueeze(dim=1))

            current_Q = (current_Q1+current_Q2+current_Q3+current_Q4+current_Q5)/5

            # Compute q_eval loss
            Q_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar(
                'Loss/Q_value', Q_loss, global_step=self.num_q_update_iteration)
            # Optimize the critic
            self.q_eval_optimizer1.zero_grad()
            self.q_eval_optimizer2.zero_grad()
            self.q_eval_optimizer3.zero_grad()
            self.q_eval_optimizer4.zero_grad()
            self.q_eval_optimizer5.zero_grad()
            Q_loss.backward()
            self.q_eval_optimizer1.step()
            self.q_eval_optimizer2.step()
            self.q_eval_optimizer3.step()
            self.q_eval_optimizer4.step()
            self.q_eval_optimizer5.step()

            # Update the frozen target models
            # 每个iteration对Q参数进行软替换
            for param, target_param in zip(self.q_eval1.parameters(), self.q_next1.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(self.q_eval2.parameters(), self.q_next2.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(self.q_eval3.parameters(), self.q_next3.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(self.q_eval4.parameters(), self.q_next4.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(self.q_eval5.parameters(), self.q_next5.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data)
            self.num_q_update_iteration += 1

    def save(self):
        torch.save(self.q_eval1.state_dict(), directory + 'q_eval1.pth')
        torch.save(self.q_eval2.state_dict(), directory + 'q_eval2.pth')
        torch.save(self.q_eval3.state_dict(), directory + 'q_eval3.pth')
        torch.save(self.q_eval3.state_dict(), directory + 'q_eval4.pth')
        torch.save(self.q_eval3.state_dict(), directory + 'q_eval5.pth')
        torch.save(self.q_next1.state_dict(), directory + 'q_next1.pth')
        torch.save(self.q_next2.state_dict(), directory + 'q_next2.pth')
        torch.save(self.q_next3.state_dict(), directory + 'q_next3.pth')
        torch.save(self.q_next3.state_dict(), directory + 'q_next4.pth')
        torch.save(self.q_next3.state_dict(), directory + 'q_next5.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.q_eval1.load_state_dict(torch.load(directory + 'q_eval1.pth', map_location='cpu'))
        self.q_eval2.load_state_dict(torch.load(directory + 'q_eval2.pth', map_location='cpu'))
        self.q_eval3.load_state_dict(torch.load(directory + 'q_eval3.pth', map_location='cpu'))
        self.q_eval4.load_state_dict(torch.load(directory + 'q_eval4.pth', map_location='cpu'))
        self.q_eval5.load_state_dict(torch.load(directory + 'q_eval5.pth', map_location='cpu'))
        self.q_next1.load_state_dict(torch.load(directory + 'q_next1.pth', map_location='cpu'))
        self.q_next2.load_state_dict(torch.load(directory + 'q_next2.pth', map_location='cpu'))
        self.q_next3.load_state_dict(torch.load(directory + 'q_next3.pth', map_location='cpu'))
        self.q_next4.load_state_dict(torch.load(directory + 'q_next4.pth', map_location='cpu'))
        self.q_next5.load_state_dict(torch.load(directory + 'q_next5.pth', map_location='cpu'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def main():
    agent = DQN(state_dim, action_dim, max_action)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            env.reset()
            env.resend()
            for t in count():
                shift = agent.select_action(state)
                # print(action)
                next_state, reward, done = env.step(np.float32(shift))
                ep_r += reward
                if args.render and i >= args.render_interval:
                    env.render()
                    env.clock.tick(10)
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
        #if args.load:
            #agent.load()
        for i in range(args.max_episode):
            env.reset()
            state = env.resend()
            # ep_r = 0
            for t in count():

                action = agent.select_action(state)
                shift = max_action - action * 2 * max_action/args.n_actions
                # issue 3 add noise to action
                shift = (shift + np.random.normal(0, args.exploration_noise, size=action_dim)).clip(
                    -max_action, max_action)
                # print(action)
                # todo
                # 这里的shift只有一个导弹的旋转角度
                # 但是实际应该输入导弹组的旋转角度
                # 根本原因是一次Q_network只能得到一个导弹的动作Q值
                # 而DDPG里的actor采用策略梯度直接得到每个agent的动作
                next_state, reward, done = env.step(shift)
                ep_r += reward
                # if args.render and i >= args.render_interval:
                #     env.render()
                #     env.clock.tick(10)
                agent.replay_buffer.push(
                    (state, next_state, shift, reward, np.float(done)))
                # if (i+1) % 10 == 0:
                #     print('Episode {},  The memory size is {} '.format(i, len(agent.replay_buffer.storage)))

                state = next_state
                if done or t >= args.max_length_of_trajectory:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    if i % args.print_log == 0:
                        print("Ep_i:  {}, the ep_r is:  {:0.2f}, the step is:  {}.".format(i, ep_r, t))
                    ep_r = 0
                    break

            if i % args.log_interval == 0:
                agent.save()
            # memory检测,满的时候开始训练
            if i % 10 == 0:
                print('Episode {},  The memory size is {} '.format(i, len(agent.replay_buffer.storage)))
            if len(agent.replay_buffer.storage) >= args.capacity - 1:
                agent.update()

    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    main()
