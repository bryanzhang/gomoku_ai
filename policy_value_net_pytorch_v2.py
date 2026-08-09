#! /usr/bin/python3

# v2 网络: ResNet 版策略价值网络(v4 报告路线图第 2 步)。
# v1 原 3 层 conv 网络保留在 policy_value_net_pytorch.py 中不变, 旧 .model/.pt 权重照常加载;
# 本文件与 v1 的输入输出格式完全一致(log_softmax 策略 + tanh 价值), C++ 侧吃 torchscript
# trace, 换结构不需要改任何搜索代码。v1/v2 权重互不通用, load_net_any_arch 会按权重内容
# 自动识别结构(并推断 v2 的 blocks/channels)。

import copy
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

    # 导出用副本: 深拷贝训练网络, 把每个 conv(无bias)+BN 对折叠成带 bias 的 conv、
    # BN 换成 Identity, 效果等同 torch.jit.freeze 的 BN 折叠(数值 ~1e-6 一致)。
    # 折叠只发生一次; 之后每次导出前 _refresh_export_net 按训练网络的当前权重重算
    # 折叠值, traced 模块与副本共享参数存储, 无需重新 trace/freeze(那会每调一次
    # 泄漏数 MB, 见基类 save_model_with_torchscript 的注释)。
    def _conv_bn_pairs(self, net):
        pairs = [(net.conv1, net.bn1)]
        for blk in net.blocks:
            pairs.append((blk.conv1, blk.bn1))
            pairs.append((blk.conv2, blk.bn2))
        pairs.append((net.act_conv1, net.act_bn))
        pairs.append((net.val_conv1, net.val_bn))
        return pairs

    def _build_export_net(self):
        if getattr(self, '_export_folded_net', None) is None:
            folded = copy.deepcopy(self.policy_value_net).eval()
            # 给每个无 bias 的 conv 补上 bias 参数(折叠后 BN 的仿射项进这里)
            for conv, _ in self._conv_bn_pairs(folded):
                conv.bias = nn.Parameter(torch.zeros(conv.out_channels))
            # BN 已折叠进 conv, 换成 Identity 使 forward 不再出现 BN
            folded.bn1 = nn.Identity()
            for blk in folded.blocks:
                blk.bn1 = nn.Identity()
                blk.bn2 = nn.Identity()
            folded.act_bn = nn.Identity()
            folded.val_bn = nn.Identity()
            self._export_folded_net = folded
            self._refresh_export_net()
        return self._export_folded_net

    # 把训练网络当前的 conv+BN 折叠进导出副本: W' = W * γ/√(σ²+ε), b' = β - μγ/√(σ²+ε)
    def _refresh_export_net(self):
        src_net = self.policy_value_net
        dst_net = self._export_folded_net
        with torch.no_grad():
            for (s_conv, s_bn), (d_conv, _) in zip(self._conv_bn_pairs(src_net),
                                                   self._conv_bn_pairs(dst_net)):
                scale = s_bn.weight / torch.sqrt(s_bn.running_var + s_bn.eps)
                d_conv.weight.copy_(s_conv.weight * scale.view(-1, 1, 1, 1))
                d_conv.bias.copy_(s_bn.bias - s_bn.running_mean * scale)
            # 非折叠参数(fc 层)直接同步
            for name in ('act_fc1', 'val_fc1', 'val_fc2'):
                getattr(dst_net, name).load_state_dict(getattr(src_net, name).state_dict())

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
