"""DLP DQN Lab"""
__author__ = "chengscott"
__copyright__ = "Copyright 2020, NCTU CGI Lab"
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import wrap_deepmind, make_atari


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.size = len(self.memory)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return self.size


class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.0
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(
            self._behavior_net.parameters(), lr=args.lr, eps=1.5e-4
        )

        ## TODO ##
        """Initialize replay buffer"""
        self._memory = ReplayMemory(args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        if random.random() < epsilon:
            return action_space.sample()
        with torch.no_grad():
            state = np.transpose(state, (2, 0, 1))  # (H, W, C) -> (C, H, W)

            if state.shape[0] < 4:
                # pad state with last frame if state is not enough
                # (C, H, W) -> (4, C, H, W)
                state = np.concatenate(
                    [state for _ in range(4 - state.shape[0] + 1)], axis=0
                )

            state = np.expand_dims(state, axis=0)
            st = torch.from_numpy(state).to(self.device)
            return self._behavior_net(st).max(dim=1)[1].item()

    def append(self, state, action, reward, next_state, done):
        ## TODO ##
        """Push a transition into replay buffer"""
        self._memory.push(state, action, reward, next_state, done)

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size)
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        next_state = torch.tensor(
            np.array(next_state), device=self.device, dtype=torch.float32
        )
        done = torch.tensor(done, device=self.device, dtype=torch.float32)

        # Compute Q-values
        state = state.permute(0, 3, 1, 2)
        next_state = next_state.permute(0, 3, 1, 2)
        q_values = self._behavior_net(state)
        q_values = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Compute the target Q-values
        with torch.no_grad():
            q_next = self._target_net(next_state).max(dim=1)[
                0
            ]  # max 函數返回最大值，所以這裡取的是 "最佳" 下一步動作的 Q-值
            q_target = reward + gamma * q_next * (1 - done)

        # Compute and minimize the loss
        criterion = nn.MSELoss()
        loss = criterion(q_values, q_target)

        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        """update target network by copying from behavior network"""
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    "behavior_net": self._behavior_net.state_dict(),
                    "target_net": self._target_net.state_dict(),
                    "optimizer": self._optimizer.state_dict(),
                },
                model_path,
            )
        else:
            torch.save(
                {
                    "behavior_net": self._behavior_net.state_dict(),
                },
                model_path,
            )

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model["behavior_net"])
        if checkpoint:
            self._target_net.load_state_dict(model["target_net"])
            self._optimizer.load_state_dict(model["optimizer"])


def train(args, agent: DQN, writer: SummaryWriter):
    print("Start Training")
    env_raw = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env_raw, frame_stack=True, clip_rewards=True, scale=False)
    action_space = env.action_space
    total_steps, epsilon = 0, 1.0
    ewma_reward = 0
    best_reward = -np.inf

    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        state, reward, done, _ = env.step(1)  # fire first !!!
        state = np.array(state)
        for t in itertools.count(start=1):
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)

            # execute action
            next_state, reward, done, _ = env.step(action)

            # store transition (TODO part)
            agent.append(state, action, reward, next_state, done)

            state = next_state

            if total_steps >= args.warmup:
                agent.update(total_steps)

            total_reward += reward

            if total_steps % args.eval_freq == 0:
                """You can write another evaluate function, or just call the test function."""
                evaludate(args, agent, writer, total_steps)
                agent.save(args.model + "dqn_" + str(total_steps) + ".pt")

            total_steps += 1

            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                if ewma_reward > best_reward:
                    best_reward = ewma_reward
                    agent.save(args.model + "dqn_breakout_best.pt")

                writer.add_scalar("Train/Episode Reward", total_reward, episode)
                writer.add_scalar("Train/Ewma Reward", ewma_reward, episode)
                print(
                    "Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}".format(
                        total_steps, episode, t, total_reward, ewma_reward, epsilon
                    )
                )
                break
    env.close()


@torch.no_grad()
def evaludate(args, agent: DQN, writer: SummaryWriter, current_steps: int):
    env_raw = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env_raw)
    action_space = env.action_space
    e_rewards = []

    for i in range(args.test_episode):
        state = env.reset()
        e_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, args.test_epsilon, action_space)
            state, reward, done, _ = env.step(action)
            e_reward += reward

        print("episode {}: {:.2f}".format(i + 1, e_reward))
        e_rewards.append(e_reward)

    env.close()
    avg_reward = float(sum(e_rewards)) / float(args.test_episode)
    writer.add_scalar("Evaluate/Average Reward", avg_reward, current_steps)


@torch.no_grad()
def test(args, agent: DQN, writer: SummaryWriter):
    print("Start Testing")
    env_raw = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env_raw)
    action_space = env.action_space
    e_rewards = []

    for i in range(args.test_episode):
        state = env.reset()
        e_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, args.test_epsilon, action_space)
            state, reward, done, _ = env.step(action)
            e_reward += reward

        print("episode {}: {:.2f}".format(i + 1, e_reward))
        e_rewards.append(e_reward)

    env.close()
    avg_reward = float(sum(e_rewards)) / float(args.test_episode)
    print("Average Reward: {:.2f}".format(avg_reward))


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-m", "--model", default="ckpt/")
    parser.add_argument("--logdir", default="log/dqn")
    # train
    parser.add_argument("--warmup", default=20000, type=int)
    parser.add_argument("--episode", default=50000, type=int)
    parser.add_argument("--capacity", default=100000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.0000625, type=float)
    parser.add_argument("--eps_decay", default=0.995, type=float)
    parser.add_argument("--eps_min", default=0.1, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--freq", default=4, type=int)
    parser.add_argument("--target_freq", default=10000, type=int)
    parser.add_argument("--eval_freq", default=200000, type=int)
    # test
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("-tmp", "--test_model_path", default="ckpt/dqn_breakout.pt")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--test_episode", default=10, type=int)
    parser.add_argument("--seed", default=20230422, type=int)
    parser.add_argument("--test_epsilon", default=0.01, type=float)
    args = parser.parse_args()

    import os

    os.makedirs(args.model, exist_ok=True)

    ## main ##
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load(args.test_model_path)
        test(args, agent, writer)
    else:
        train(args, agent, writer)


if __name__ == "__main__":
    main()
