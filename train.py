#! /usr/bin/python3

from policy_value_net_pytorch import PolicyValueNet
import random
from game import Game
from player import AlphaZeroPlayer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
import os
import sys
import time
from collections import deque, defaultdict

class TrainPipeline():
    def __init__(self, init_model=None):
        self.board_width = 11
        self.board_height = 11
        self.n_in_row = 5
        self.game_batch_num = 1500
        self.play_batch_size = 1
        self.batch_size = 512
        # 每多少个 batch 评估一次(play_batch_size=1 时即每 100 局自对弈评估一次)
        self.check_freq = 100000
        self.eval_n_games = 20  # 每次评估的对局数, 必须是偶数以保证黑白各执一半
        # 自对弈前 temp_moves 手用 temperature=1.0 探索, 之后 τ→0 走最强手
        self.temp_moves = 15
        self.save_freq = 10  # 每多少个 batch 落盘一次 checkpoint
        self.n_playout = 1000
        self.kl_targ = 0.02
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        self.game = Game()
        self.tmp_model_path = "/tmp/gomoku_model.pt"
        self.ckpt_path = "./current_policy.ckpt"
        # 评估相关: baseline 就是"上一次评估时的版本", 以它为锚点做 Elo 增量
        self.baseline_model_path = "./eval_baseline.model"  # 上个版本的权重(state_dict)
        self.eval_cur_ts_path = "/tmp/gomoku_model_eval_cur.pt"    # 当前版本 torchscript
        self.eval_prev_ts_path = "/tmp/gomoku_model_eval_prev.pt"  # 上个版本 torchscript
        self.elo = 0.0       # 相对 Elo, 以第一个 baseline 版本为 0 分锚点
        self.best_elo = 0.0
        self.mcts_player = AlphaZeroPlayer(self.n_playout, self.tmp_model_path, 16, 5.0, True)
        self.data_buffer = deque(maxlen=10000)
        self.epochs = 5 # num of train_steps for each update
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0 # adaptively adjust the learning rate based on KL
        self.start_batch = 0
        # 从 checkpoint 恢复训练进度
        extra = self.policy_value_net.extra_state
        if extra:
            self.start_batch = extra.get('batch', 0)
            self.lr_multiplier = extra.get('lr_multiplier', 1.0)
            self.elo = extra.get('elo', 0.0)
            self.best_elo = extra.get('best_elo', self.elo)
            print(f"Resumed from batch {self.start_batch}, lr_multiplier={self.lr_multiplier:.3f}, elo={self.elo:.1f}", file=sys.stderr)
        # 首次训练时把初始模型存为 baseline, 这样第一次评估就有"上个版本"可以对弈
        if not os.path.exists(self.baseline_model_path):
            self.snapshot_baseline()

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

    # 把得分率换算成相对 Elo 差: diff = 400 * log10(S / (1 - S))
    # 全胜/全负时 log 会发散, 用 1/(2*(n+1)) 做截断, 相当于给两边各加半局的先验
    @staticmethod
    def score_to_elo_diff(score, n_games):
        eps = 1.0 / (2.0 * (n_games + 1))
        s = min(max(score, eps), 1.0 - eps)
        return 400.0 * math.log10(s / (1.0 - s))

    def snapshot_baseline(self):
        self.policy_value_net.save_model(self.baseline_model_path)

    # 只和上一个评估版本(baseline)对弈, 用 Elo 衡量进步
    # 返回 (score, elo_diff, ci_low, ci_high); 没有 baseline 时返回 None
    def policy_evaluate(self, n_games=None):
        n_games = n_games or self.eval_n_games
        if not os.path.exists(self.baseline_model_path):
            self.snapshot_baseline()
            print("No baseline yet, snapshot current model as baseline.", file=sys.stderr)
            return None

        start_time = time.time()
        # 双方各自导出一份 torchscript, C++ 侧按 model_path 各自加载, 互不影响
        self.policy_value_net.save_model_with_torchscript(self.eval_cur_ts_path)
        baseline_net = PolicyValueNet(self.board_width, self.board_height, model_file=self.baseline_model_path)
        baseline_net.save_model_with_torchscript(self.eval_prev_ts_path)

        current_player = AlphaZeroPlayer(self.n_playout, self.eval_cur_ts_path, 16, 5.0, True)
        baseline_player = AlphaZeroPlayer(self.n_playout, self.eval_prev_ts_path, 16, 5.0, True)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            # 每局都要重建搜索树, 否则下一局的棋盘状态对不上
            current_player.reset()
            baseline_player.reset()
            if (i % 2) == 0:
                winner, _ = self.game.start_play(current_player, baseline_player)
            else:
                winner, _ = self.game.start_play(baseline_player, current_player)
                if winner != -1:
                    winner = 1 - winner
            win_cnt[winner] += 1

        wins, losses, draws = win_cnt[0], win_cnt[1], win_cnt[-1]
        score = (wins + 0.5 * draws) / n_games  # 当前版本的得分率
        elo_diff = self.score_to_elo_diff(score, n_games)
        # 正态近似的 95% 置信区间, 用来判断这次提升是不是噪声
        # 用截断后的得分率算标准误, 避免全胜/全负时区间退化成一个点
        eps = 1.0 / (2.0 * (n_games + 1))
        s_clamped = min(max(score, eps), 1.0 - eps)
        se = math.sqrt(s_clamped * (1.0 - s_clamped) / n_games)
        ci_low = self.score_to_elo_diff(score - 1.96 * se, n_games)
        ci_high = self.score_to_elo_diff(score + 1.96 * se, n_games)
        # baseline 的 Elo 就是更新前的 self.elo, 所以直接叠加即可
        self.elo += elo_diff
        execution_time = time.time() - start_time
        print(f"[Eval vs prev] games:{n_games}, win:{wins}, lose:{losses}, draw:{draws}, "
              f"score:{score:.3f}, elo_diff:{elo_diff:+.1f} (95%CI {ci_low:+.0f}~{ci_high:+.0f}), "
              f"elo:{self.elo:.1f}, eval time: {execution_time:.1f} seconds", file=sys.stderr)
        # 当前版本成为下一轮的"上个版本"
        self.snapshot_baseline()
        return score, elo_diff, ci_low, ci_high

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
                                              lr_multiplier=self.lr_multiplier,
                                              elo=self.elo,
                                              best_elo=self.best_elo)
        self.policy_value_net.save_model('./current_policy.model')
        print(f"Checkpoint saved at batch {batch_no + 1} -> {self.ckpt_path}", file=sys.stderr)

    def run(self):
        writer = SummaryWriter("./gomoku_experiments")
        i = self.start_batch
        try:
            for i in range(self.start_batch, self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print(f"Batch i:#{i+1}, episolde_len:{self.episode_len}", file=sys.stderr)
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_value_update(i)
                    writer.add_scalar('Loss/Train', loss, i)
                    writer.add_scalar('Entropy/Train', entropy, i)
                # 先评估再存档, 保证这一轮的 elo 能被写进 checkpoint
                if (i + 1) % self.check_freq == 0:
                    result = self.policy_evaluate()
                    if result is not None:
                        score, elo_diff, ci_low, ci_high = result
                        writer.add_scalar('Eval/Elo', self.elo, i)
                        writer.add_scalar('Eval/EloDiffVsPrev', elo_diff, i)
                        writer.add_scalar('Eval/ScoreVsPrev', score, i)
                        if self.elo > self.best_elo:
                            self.best_elo = self.elo
                            self.policy_value_net.save_model('./best_policy.model')
                            print(f"New best policy! elo={self.elo:.1f}", file=sys.stderr)
                        elif ci_high < 0:
                            print(f"Warning: model regressed vs previous version (elo_diff={elo_diff:+.1f})", file=sys.stderr)
                if (i + 1) % self.save_freq == 0:
                    self.save_checkpoint(i)
        except KeyboardInterrupt:
            print("\nInterrupted, saving checkpoint...", file=sys.stderr)
        finally:
            self.save_checkpoint(i)
            writer.close()

if __name__ == '__main__':
    init_model = sys.argv[1] if len(sys.argv) > 1 else None
    training_pipeline = TrainPipeline(init_model=init_model)
    training_pipeline.run()
