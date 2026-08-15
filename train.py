#! /usr/bin/python3

# 训练用 v2(ResNet) 网络; v1(3conv) 保留在 policy_value_net_pytorch.py 供旧权重加载
import os
# 必须在 import torch(由 policy_value_net_pytorch_v2 间接引入)之前设置:
# 本机 NNPACK 初始化失败, c10 每次 conv 都打一条 WARNING 刷屏, 提到 ERROR 级屏蔽。
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
from policy_value_net_pytorch_v2 import PolicyValueNetV2 as PolicyValueNet, load_net_any_arch
import argparse
import random
from game import Game
from player import AlphaZeroPlayer, PureMCTSPlayer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys
import tempfile
import time
from collections import deque

class TrainPipeline():
    def __init__(self, init_model=None, board_size=11, n_playout=1000, temp_moves=15,
                 game_batch_num=10000, eval_freq=1000, eval_games=20,
                 opp_model='', opp_simulations=500000, opp_c_puct=2.0, opp_reuse_states=True,
                 num_blocks=3, channels=64):
        self.board_width = board_size
        self.board_height = board_size
        self.board_size = board_size
        self.n_in_row = 5
        self.game_batch_num = game_batch_num
        self.play_batch_size = 1
        self.batch_size = 512
        # 每多少局自对弈(play_batch_size=1 时每 batch 一局)做一次评估
        self.eval_freq = eval_freq
        self.eval_n_games = eval_games  # 每次评估的对局数, 建议偶数以保证黑白各执一半
        # 自对弈前 temp_moves 手用 temperature=1.0 探索, 之后 τ→0 走最强手
        self.temp_moves = temp_moves
        self.save_freq = 10  # 每多少个 batch 落盘一次 checkpoint
        self.n_playout = n_playout
        self.kl_targ = 0.02
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model,
                                               num_blocks=num_blocks, channels=channels)
        self.game = Game(self.board_width, self.board_height)
        # 所有落盘/临时路径都带棋盘尺寸后缀(11x11 保持历史路径不变),
        # 防止不同尺寸的训练任务互相覆盖 checkpoint、抢同一个 torchscript 临时文件。
        suffix = "" if board_size == 11 else f"_{board_size}x{board_size}"
        self.tmp_model_path = f"/tmp/gomoku_model{suffix}.pt"
        self.ckpt_path = f"./current_policy{suffix}.ckpt"
        self.model_path = f"./current_policy{suffix}.model"
        self.log_dir = f"./gomoku_experiments{suffix}"
        # 评估(类似 elo.py): 与固定对手交替执黑对弈, 只记录指标(overall/black/white
        # score, avg steps), 不做早停或选模。对手默认 model-free 纯 MCTS;
        # opp_model 非空(.model/.ckpt/.pt)时加载为 AlphaZero 对手。
        self.eval_cur_ts_path = f"/tmp/gomoku_model_eval_cur{suffix}.pt"  # 当前版本 torchscript
        self.opp_model = opp_model
        self.opp_simulations = opp_simulations
        self.opp_c_puct = opp_c_puct
        self.opp_reuse_states = opp_reuse_states
        self.eval_opp_player = None  # 首次评估时构建, 之后复用(对手不随训练变化)
        # 每次评估时把当时的模型存一份快照, 训练完后可任意挑选用于测试
        self.eval_snapshot_dir = f"./{board_size}x{board_size}_snapshots"
        self.mcts_player = AlphaZeroPlayer(board_size, self.n_playout, self.tmp_model_path, 16, 5.0, True)
        # v4 路线图第 2 步: buffer 10000 -> 50000。8 倍增广后每局 ~400 条, 10000 只装
        # ~25 局, 网络一直在"最近 25 局"上原地踏步; 50000 约 125 局窗口(~250MB 内存)。
        self.data_buffer = deque(maxlen=10000)
        self.epochs = 5 # num of train_steps for each update
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0 # adaptively adjust the learning rate based on KL
        self.start_batch = 0
        # 从 checkpoint 恢复训练进度(旧 checkpoint 里的 elo/best_elo 字段已废弃, 忽略)
        extra = self.policy_value_net.extra_state
        if extra:
            self.start_batch = extra.get('batch', 0)
            self.lr_multiplier = extra.get('lr_multiplier', 1.0)
            print(f"Resumed from batch {self.start_batch}, lr_multiplier={self.lr_multiplier:.3f}", file=sys.stderr)

    def get_augumented_data(self, play_data):
        # 旋转和翻转得到更多样本,共产生8倍样本
        # TODO(junhaozhang): 可以有一半的样本再黑白棋反转，额外产生4倍样本
        # NOTE(junhaozhang): state 平面按 [y][x] 索引, mcts_prob 平铺下标 idx = y*W + x,
        # 二者布局一致, 因此对状态和概率图施加完全相同的几何变换即可, 不需要额外的 flipud。
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(mcts_prob.reshape(self.board_height, self.board_width), i)
                extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))

                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games = 1):
        self.policy_value_net.save_model_with_torchscript(self.tmp_model_path)
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp_moves=self.temp_moves)
            play_data = list(play_data)
            self.episode_len = len(play_data)
            play_data = self.get_augumented_data(play_data)
            self.data_buffer.extend(play_data)

    # 构建评估对手。纯 MCTS 不需要导出模型; AlphaZero 对手的非 .pt 模型
    # (.model/.ckpt) 先按内容自动识别 v1/v2 结构并导出 torchscript。
    def build_eval_opponent(self):
        if not self.opp_model:
            desc = (f"PureMCTS(sims={self.opp_simulations}, c_puct={self.opp_c_puct}, "
                    f"reuse_states={self.opp_reuse_states})")
            return PureMCTSPlayer(self.board_size, self.opp_simulations, 16, self.opp_c_puct,
                                  self.opp_reuse_states), desc
        ts_path = self.opp_model
        if not ts_path.endswith('.pt'):
            net = load_net_any_arch(self.board_width, self.board_height, self.opp_model)
            fd, ts_path = tempfile.mkstemp(prefix='eval_opp_', suffix='.pt')
            os.close(fd)
            net.save_model_with_torchscript(ts_path)
            print(f"[Eval] exported opponent torchscript {self.opp_model} -> {ts_path}", file=sys.stderr)
        desc = (f"AlphaZero(model={self.opp_model}, sims={self.opp_simulations}, "
                f"c_puct={self.opp_c_puct}, reuse_states={self.opp_reuse_states})")
        return AlphaZeroPlayer(self.board_size, self.opp_simulations, ts_path, 16, self.opp_c_puct,
                               self.opp_reuse_states), desc

    # 固定对手评估(类似 elo.py): 当前模型与对手交替执黑, 各执黑一半局数。
    # 返回 (overall_score, black_score, white_score, avg_steps), 都是当前模型视角。
    def policy_evaluate(self, games_played):
        start_time = time.time()
        # NOTE(junhaozhang): 必须用独立的 Game 实例! start_self_play 依赖 self.game 的
        # 棋盘在上局结束时是空的, 若评估复用 self.game, 评估终局会残留在棋盘上,
        # 导致下一次自对弈第一手 StateEquals 失配。
        eval_game = Game(self.board_width, self.board_height)
        # 保存当时模型的快照(state_dict), 与本次评估所见权重一致, 便于事后挑选测试
        os.makedirs(self.eval_snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(self.eval_snapshot_dir, f'policy_game_{games_played}.model')
        self.policy_value_net.save_model(snapshot_path)
        # 当前模型每次评估都要重新导出(权重在训练中不断变化)
        self.policy_value_net.save_model_with_torchscript(self.eval_cur_ts_path)
        current_player = AlphaZeroPlayer(self.board_size, self.n_playout, self.eval_cur_ts_path, 16, 5.0, True)
        if self.eval_opp_player is None:
            self.eval_opp_player, self.eval_opp_desc = self.build_eval_opponent()
            print(f"[Eval] opponent: {self.eval_opp_desc}", file=sys.stderr)
        opponent = self.eval_opp_player

        n = self.eval_n_games
        stats = {'black': [0, 0, 0], 'white': [0, 0, 0]}  # 当前模型视角: [胜, 负, 和]
        total_steps = 0
        for i in range(n):
            # 每局重置双方搜索树(保留线程池), 否则下一局棋盘状态对不上
            current_player.reset()
            opponent.reset()
            if i % 2 == 0:
                winner, steps = eval_game.start_play(current_player, opponent)
                key = 'black'
            else:
                winner, steps = eval_game.start_play(opponent, current_player)
                if winner != -1:
                    winner = 1 - winner  # 换算回当前模型视角: 0=胜, 1=负, -1=和
                key = 'white'
            stats[key][0 if winner == 0 else (1 if winner == 1 else 2)] += 1
            total_steps += steps

        wins = stats['black'][0] + stats['white'][0]
        losses = stats['black'][1] + stats['white'][1]
        draws = stats['black'][2] + stats['white'][2]
        overall_score = (wins + 0.5 * draws) / n
        # n 为奇数时黑白局数不相等, 分开归一
        black_score = (stats['black'][0] + 0.5 * stats['black'][2]) / sum(stats['black'])
        white_score = (stats['white'][0] + 0.5 * stats['white'][2]) / sum(stats['white'])
        avg_steps = total_steps / n
        print(f"[Eval] games:{n}, overall_score:{overall_score:.3f}, black_score:{black_score:.3f}, "
              f"white_score:{white_score:.3f}, avg_steps:{avg_steps:.1f} "
              f"(win:{wins}, lose:{losses}, draw:{draws}), snapshot: {snapshot_path}, "
              f"eval time: {time.time() - start_time:.1f} seconds", file=sys.stderr)
        return overall_score, black_score, white_score, avg_steps

    def policy_value_update(self, seq_no):
        start_time = time.time()
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        # PPO
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        execution_time = time.time() - start_time
        print(f"BatchNo.{seq_no+1}, KL:{kl:.5f}, lr_+multiplier: {self.lr_multiplier:.3f}, loss:{loss}, entropy:{entropy}, explained_var_old: {explained_var_old:.3f}, explained_var_new: {explained_var_new:.3f}, learn time: {execution_time:.3f} seconds", file=sys.stderr)
        return loss, entropy

    def save_checkpoint(self, batch_no):
        self.policy_value_net.save_checkpoint(self.ckpt_path,
                                              batch=batch_no + 1,
                                              lr_multiplier=self.lr_multiplier)
        self.policy_value_net.save_model(self.model_path)
        print(f"Checkpoint saved at batch {batch_no + 1} -> {self.ckpt_path}", file=sys.stderr)

    def run(self):
        writer = SummaryWriter(self.log_dir)
        i = self.start_batch
        try:
            for i in range(self.start_batch, self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print(f"Batch i:#{i+1}, episolde_len:{self.episode_len}", file=sys.stderr)
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_value_update(i)
                    writer.add_scalar('Loss/Train', loss, i)
                    writer.add_scalar('Entropy/Train', entropy, i)
                # 固定对手评估, 只记录指标到 tensorboard(与 train curve 对照看), 不做早停
                if (i + 1) % self.eval_freq == 0:
                    overall, black_score, white_score, avg_steps = self.policy_evaluate(i + 1)
                    writer.add_scalar('Eval/OverallScore', overall, i)
                    writer.add_scalar('Eval/BlackScore', black_score, i)
                    writer.add_scalar('Eval/WhiteScore', white_score, i)
                    writer.add_scalar('Eval/AvgSteps', avg_steps, i)
                if (i + 1) % self.save_freq == 0:
                    self.save_checkpoint(i)
        except KeyboardInterrupt:
            print("\nInterrupted, saving checkpoint...", file=sys.stderr)
        finally:
            self.save_checkpoint(i)
            writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AlphaZero gomoku training pipeline (v2 ResNet).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('init_model', nargs='?', default=None,
                        help='初始模型(.model/.ckpt), 缺省从零开始训练')
    parser.add_argument('--board-size', type=int, default=11, choices=[8, 11, 15],
                        help='棋盘边长(8/11/15, 对应 gomoku_ai 的 C++ 绑定); 15x15 的 checkpoint/'
                             '快照/tensorboard 目录会自动带 _15x15 后缀, 不会覆盖 11x11 的产物')
    parser.add_argument('--num-blocks', type=int, default=3,
                        help='策略值网络(v2 ResNet)的残差块数, 大棋盘可适当加大')
    parser.add_argument('--channels', type=int, default=64,
                        help='策略值网络(v2 ResNet)的通道数, 大棋盘可适当加大')
    parser.add_argument('--n-playout', type=int, default=1000,
                        help='自对弈与评估时当前模型每手的 MCTS 模拟次数')
    parser.add_argument('--temp-moves', type=int, default=15,
                        help='自对弈前 N 手用 temperature=1.0 探索, 之后 τ→0 走最强手')
    parser.add_argument('--games', type=int, default=10000, help='总自对弈局数')
    parser.add_argument('--eval-freq', type=int, default=1000, help='每隔多少局评估一次')
    parser.add_argument('--eval-games', type=int, default=20,
                        help='每次评估的对局数, 建议偶数以保证黑白各执一半')
    parser.add_argument('--opp-model', default='',
                        help='评估对手模型(.model/.ckpt/.pt, v1/v2 自动识别); 置空为 model-free 纯 MCTS')
    parser.add_argument('--opp-simulations', type=int, default=500000,
                        help='评估对手每手 MCTS 模拟次数')
    parser.add_argument('--opp-c-puct', type=float, default=2.0, help='评估对手 PUCT 常数')
    parser.add_argument('--opp-reuse-states', action=argparse.BooleanOptionalAction, default=True,
                        help='评估对手是否复用搜索树')
    args = parser.parse_args()
    training_pipeline = TrainPipeline(init_model=args.init_model,
                                      board_size=args.board_size,
                                      n_playout=args.n_playout,
                                      temp_moves=args.temp_moves,
                                      game_batch_num=args.games,
                                      eval_freq=args.eval_freq,
                                      eval_games=args.eval_games,
                                      opp_model=args.opp_model,
                                      opp_simulations=args.opp_simulations,
                                      opp_c_puct=args.opp_c_puct,
                                      opp_reuse_states=args.opp_reuse_states,
                                      num_blocks=args.num_blocks,
                                      channels=args.channels)
    training_pipeline.run()
