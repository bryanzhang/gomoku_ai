#! /usr/bin/python3

# v2 network: ResNet-style policy-value network.
# The original v1 3-conv network is kept unchanged in
# policy_value_net_pytorch.py, and old .model/.pt weights still load as
# before; this file has exactly the same input/output format as v1
# (log_softmax policy + tanh value), and the C++ side consumes a torchscript
# trace, so switching architectures requires no search-code changes. v1/v2
# weights are not interchangeable; load_net_any_arch auto-detects the
# architecture from the weight content (and infers v2's blocks/channels).

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
    # Default 3 blocks / 64 channels (~300k parameters).
    # NOTE(junhaozhang): shrinking the config further yields diminishing
    # returns -- the bottleneck is network forward passes inside MCTS, but
    # each worker thread keeps its own torch module copy.
    def __init__(self, board_width, board_height, model_file=None, num_blocks=3, channels=64):
        super().__init__(board_width, board_height)  # build the v1 placeholder first, then replace with the v2 architecture
        self.policy_value_net = NetV2(board_width, board_height, num_blocks, channels)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)
        self.extra_state = {}
        if model_file:
            self.load_model(model_file)

    # NOTE(junhaozhang): with BatchNorm, inference/export must run under
    # eval() (using running stats) and training must run under train()
    # (running stats update per batch); mixing them up produces inconsistent
    # results.
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

    # Export copy: deep-copy the training net, fold every conv (bias-free) +
    # BN pair into a conv with bias, and replace BN with Identity -- the
    # effect equals torch.jit.freeze's BN folding (numerically identical to
    # ~1e-6). Folding happens only once; before each later export,
    # _refresh_export_net recomputes the folded values from the training
    # net's current weights. The traced module shares parameter storage with
    # the copy, so no re-trace/freeze is needed (those leak several MB per
    # call -- see the base class save_model_with_torchscript comment).
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
            # Add a bias parameter to every bias-free conv (the folded BN
            # affine term lands here).
            for conv, _ in self._conv_bn_pairs(folded):
                conv.bias = nn.Parameter(torch.zeros(conv.out_channels))
            # BN has been folded into conv; replace with Identity so forward
            # no longer contains BN.
            folded.bn1 = nn.Identity()
            for blk in folded.blocks:
                blk.bn1 = nn.Identity()
                blk.bn2 = nn.Identity()
            folded.act_bn = nn.Identity()
            folded.val_bn = nn.Identity()
            self._export_folded_net = folded
            self._refresh_export_net()
        return self._export_folded_net

    # Fold the training net's current conv+BN into the export copy:
    # W' = W * gamma/sqrt(var+eps), b' = beta - mean*gamma/sqrt(var+eps)
    def _refresh_export_net(self):
        src_net = self.policy_value_net
        dst_net = self._export_folded_net
        with torch.no_grad():
            for (s_conv, s_bn), (d_conv, _) in zip(self._conv_bn_pairs(src_net),
                                                   self._conv_bn_pairs(dst_net)):
                scale = s_bn.weight / torch.sqrt(s_bn.running_var + s_bn.eps)
                d_conv.weight.copy_(s_conv.weight * scale.view(-1, 1, 1, 1))
                d_conv.bias.copy_(s_bn.bias - s_bn.running_mean * scale)
            # Non-folded parameters (fc layers) are synced directly.
            for name in ('act_fc1', 'val_fc1', 'val_fc2'):
                getattr(dst_net, name).load_state_dict(getattr(src_net, name).state_dict())

    def save_model_with_torchscript(self, model_file):
        was_training = self.policy_value_net.training
        self.policy_value_net.eval()
        super().save_model_with_torchscript(model_file)
        if was_training:
            self.policy_value_net.train()

    # Old weights are incompatible after an architecture change;
    # load_state_dict raises a shape mismatch -- give a clear hint.
    def load_model(self, model_file):
        try:
            super().load_model(model_file)
        except RuntimeError as e:
            raise RuntimeError(f"weights are incompatible with the current network architecture (v1/v2 or blocks/channels mismatch): {model_file}\n{e}")

# Tell whether the weights are v2 (ResNet) from the state_dict keys, and
# infer blocks/channels.
def infer_v2_arch(state_dict):
    if not any(k.startswith('blocks.') for k in state_dict):
        return None
    channels = state_dict['conv1.weight'].shape[0]
    num_blocks = len({k.split('.')[1] for k in state_dict
                      if k.startswith('blocks.') and k.split('.')[1].isdigit()})
    return num_blocks, channels

# Auto-detect v1 (3conv) / v2 (ResNet) from the weight content and load it,
# for scenarios like elo.py that do not know which version the opponent is.
def load_net_any_arch(board_width, board_height, model_file):
    ckpt = torch.load(model_file, map_location='cpu')
    state_dict = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    arch = infer_v2_arch(state_dict)
    if arch is None:
        return PolicyValueNet(board_width, board_height, model_file=model_file)
    num_blocks, channels = arch
    return PolicyValueNetV2(board_width, board_height, model_file=model_file,
                            num_blocks=num_blocks, channels=channels)
