#! /usr/bin/python3

# v2 网络: ResNet 版策略价值网络(v4 报告路线图第 2 步)。
# v1 原 3 层 conv 网络保留在 policy_value_net_pytorch.py 中不变, 旧 .model/.pt 权重照常加载;
# 本文件与 v1 的输入输出格式完全一致(log_softmax 策略 + tanh 价值), C++ 侧吃 torchscript
# trace, 换结构不需要改任何搜索代码。v1/v2 权重互不通用, load_net_any_arch 会按权重内容
# 自动识别结构(并推断 v2 的 blocks/channels)。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from policy_value_net_pytorch import PolicyValueNet

class ResBlock(nn.Module):
    # ResNet v1 block: conv-bn-relu-conv-bn + skip + relu
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class NetV2(nn.Module):
    def __init__(self, board_width, board_height, num_blocks=6, channels=128):
        super(NetV2, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # common trunk: stem + residual blocks
        self.conv1 = nn.Conv2d(4, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        # action policy layers
        self.act_conv1 = nn.Conv2d(channels, 4, kernel_size=1, bias=False)
        self.act_bn = nn.BatchNorm2d(4)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.val_bn = nn.BatchNorm2d(2)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common trunk
        x = F.relu(self.bn1(self.conv1(state_input)))
        for block in self.blocks:
            x = block(x)
        # action policy layers
        x_act = F.relu(self.act_bn(self.act_conv1(x)))
        x_act = x_act.view(-1, 4 * self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_bn(self.val_conv1(x)))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val

class PolicyValueNetV2(PolicyValueNet):
    # 默认 3 blocks / 64ch(~30 万参数): 本机实测自对弈 ~77.5s/局(6/128 为 ~210s, 2/64 为 ~70s)。
    # NOTE(junhaozhang): 继续缩配置收益锐减——瓶颈大头是 C++ 每手重新加载 16 份模型
    # (SearchBestMoveWithModel 里 ThreadLocalModels 每手构造), 而非网络前向。
    def __init__(self, board_width, board_height, model_file=None, num_blocks=3, channels=64):
        super().__init__(board_width, board_height)  # 先建 v1 占位, 随即替换为 v2 结构
        self.policy_value_net = NetV2(board_width, board_height, num_blocks, channels)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)
        self.extra_state = {}
        if model_file:
            self.load_model(model_file)

    # NOTE(junhaozhang): 带 BatchNorm 后, 推理/导出必须走 eval()(用 running stats),
    # 训练必须走 train()(running stats 随 batch 更新), 二者混用会让结果对不上。
    def policy_value(self, state_batch):
        was_training = self.policy_value_net.training
        self.policy_value_net.eval()
        with torch.no_grad():
            act_probs, value = super().policy_value(state_batch)
        if was_training:
            self.policy_value_net.train()
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        self.policy_value_net.train()
        return super().train_step(state_batch, mcts_probs, winner_batch, lr)

    def save_model_with_torchscript(self, model_file):
        was_training = self.policy_value_net.training
        self.policy_value_net.eval()
        super().save_model_with_torchscript(model_file)
        if was_training:
            self.policy_value_net.train()

    # 换网络结构后旧权重不兼容, load_state_dict 会抛 shape mismatch, 给个明确提示
    def load_model(self, model_file):
        try:
            super().load_model(model_file)
        except RuntimeError as e:
            raise RuntimeError(f"权重与当前网络结构不兼容(v1/v2 或 blocks/channels 不匹配): {model_file}\n{e}")

# 从权重 key 判断是不是 v2(ResNet), 并推断 blocks/channels
def infer_v2_arch(state_dict):
    if not any(k.startswith('blocks.') for k in state_dict):
        return None
    channels = state_dict['conv1.weight'].shape[0]
    num_blocks = len({k.split('.')[1] for k in state_dict
                      if k.startswith('blocks.') and k.split('.')[1].isdigit()})
    return num_blocks, channels

# 按权重内容自动识别 v1(3conv)/v2(ResNet) 并加载, 供 elo.py 等"不知道对方是哪个版本"的场景
def load_net_any_arch(board_width, board_height, model_file):
    ckpt = torch.load(model_file, map_location='cpu')
    state_dict = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    arch = infer_v2_arch(state_dict)
    if arch is None:
        return PolicyValueNet(board_width, board_height, model_file=model_file)
    num_blocks, channels = arch
    return PolicyValueNetV2(board_width, board_height, model_file=model_file,
                            num_blocks=num_blocks, channels=channels)
