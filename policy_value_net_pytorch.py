#! /usr/bin/python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class PolicyValueNet():
    def policy_value(self, state_batch):
        state_batch = Variable(torch.FloatTensor(state_batch))
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.numpy())
        return act_probs, value.data.numpy()

    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4 # coef of l2 penalty
        self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        self.extra_state = {}
        if model_file:
            self.load_model(model_file)

    # 既支持纯 state_dict, 也支持 save_checkpoint 产出的完整 checkpoint
    def load_model(self, model_file):
        ckpt = torch.load(model_file, map_location='cpu')
        if isinstance(ckpt, dict) and 'net' in ckpt:
            self.policy_value_net.load_state_dict(ckpt['net'])
            if 'opt' in ckpt:
                self.optimizer.load_state_dict(ckpt['opt'])
            self.extra_state = {k: v for k, v in ckpt.items() if k not in ('net', 'opt')}
        else:
            self.policy_value_net.load_state_dict(ckpt)
            self.extra_state = {}

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        state_batch = Variable(torch.FloatTensor(state_batch))
        mcts_probs = Variable(torch.FloatTensor(mcts_probs))
        winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # NOTE(junhaozhang): the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss

        # backward and optimize
        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    # 导出用的网络: v1 无 BN, 直接用训练网络本身(trace 与源共享参数存储, 权重更新
    # 自动生效)。v2 带 BN, 会覆盖为"BN 折叠进 conv 的副本"(等效 torch.jit.freeze)。
    def _build_export_net(self):
        return self.policy_value_net

    # 每次导出前把训练权重同步进导出网络; v1 共享存储, 无需任何操作
    def _refresh_export_net(self):
        pass

    # for cpp-usage
    def save_model_with_torchscript(self, model_file):
        # NOTE(junhaozhang): torch.jit.trace/freeze 每调一次会泄漏数 MB C++ 内存
        # (torch 2.8 实测, gc.collect 无效), 训练一局一导出、几千局累计十几 GB 会 OOM。
        # 因此 trace 只在首次导出时做一次, 之后复用同一 traced 模块(其参数与导出网络
        # 共享存储, 实测权重更新自动可见); BN 折叠改由 _build_export_net 在 Python 侧
        # 手工完成(v2), 效果等同 freeze 但不反复泄漏。
        if getattr(self, '_export_traced', None) is None:
            example_input = torch.randn(1, 4, self.board_width, self.board_height)
            self._export_traced = torch.jit.trace(self._build_export_net(), example_input)
        self._refresh_export_net()
        # NOTE(junhaozhang): 必须原子写! C++ 侧 ThreadLocalModels 靠 mtime 侦测模型更新
        # 并即时 reload; 直接覆写会让 worker 读到写一半的文件, torch::jit::load 抛出的
        # c10::Error 在 folly worker 里无法安全回传, 会直接 abort 整个进程(实测)。
        tmp_file = model_file + '.tmp'
        self._export_traced.save(tmp_file)
        os.replace(tmp_file, model_file)

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file):
        net_params = self.get_policy_param() # get model params
        torch.save(net_params, model_file)

    # 完整 checkpoint: 网络权重 + optimizer 状态 + 训练进度, 用于续训
    def save_checkpoint(self, ckpt_file, **extra):
        ckpt = {'net': self.policy_value_net.state_dict(),
                'opt': self.optimizer.state_dict()}
        ckpt.update(extra)
        tmp_file = ckpt_file + '.tmp'
        torch.save(ckpt, tmp_file)
        os.replace(tmp_file, ckpt_file)  # 原子替换, 避免写一半被中断导致 checkpoint 损坏
