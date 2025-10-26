#! /usr/bin/python3

from policy_value_net_pytorch import PolicyValueNet
import random
from game import Game
from player import PureMCTSPlayer, AlphaZeroPlayer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
import time

class TrainPipeline():
    def __init__(self, init_model=None):
        self.board_width = 11
        self.board_height = 11
        self.n_in_row = 5
        self.game_batch_num = 1500
        self.play_batch_size = 1
        self.batch_size = 512
        self.check_freq = 50
        self.pure_mcts_playout_num = 1500000
        self.n_playout = 50000
        self.kl_targ = 0.02
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        self.game = Game()
        self.tmp_model_path = "/tmp/gomoku_model.pt"
        self.mcts_player = AlphaZeroPlayer(self.n_playout, self.tmp_model_path, 1, 5.0, True)
        self.data_buffer = []
        self.epochs = 5 # num of train_steps for each update
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0 # adaptively adjust the learning rate based on KL

    def get_augumented_data(self, play_data):
        # 旋转和翻转得到更多样本,共产生8倍样本
        # TODO(junhaozhang): 可以有一半的样本再黑白棋反转，额外产生4倍样本
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))

                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games = 1):
        self.policy_value_net.save_model_with_torchscript(self.tmp_model_path)
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player)
            play_data = list(play_data)
            self.episode_len = len(play_data)
            play_data = self.get_augumented_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_evaluate(self, n_games = 10):
        current_player = AlphaZeroPlayer(self.policy_value_net.poplicy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        pure_mcts_player = PureMCTSPlayer(c_puct = 5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            if (i % 2) == 0:
                winner, _ = self.game.start_play(current_player, pure_mcts_player)
            else:
                winner, _ = self.game.start_play(pure_mcts_player, current_player)
                if winner != -1:
                   winner = 1 - winner
            win_cnt[winner] += 1
        win_ratio = (2.0*win_cnt[0] + win_cnt[-1]) / (2.0*n_games) # 按得分率算胜率(胜利得2分,平局得1分)
        print(f"num_playouts:{self.pure_mcts_playout_num}, win: {win_cnt[0]}, lose: {win_cnt[1]}, draw: {win_cnt[-1]}", file=sys.stderr)
        return win_ratio

    def policy_value_update(self):
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
        print(f"KL:{kl:.5f}, lr_+multiplier: {self.lr_multiplier:.3f}, loss:{loss}, entropy:{entropy}, explained_var_old: {explained_var_old:.3f}, explained_var_new: {explained_var_new:.3f}, learn time: {execution_time:.3f} seconds", file=sys.stderr)
        return loss, entropy

    def run(self):
        writer = SummaryWriter("./gomoku_experiments")
        for i in range(self.game_batch_num):
            self.collect_selfplay_data(self.play_batch_size)
            print(f"Batch i:#{i+1}, episolde_len:{self.episode_len}", file=sys.stderr)
            if len(self.data_buffer) > self.batch_size:
                loss, entropy = self.policy_value_update()
                writer.add_scalar('Loss/Train', loss, i)
                writer.add_scalar('Entropy/Train', entropy, i)
            if (i + 1) % self.check_freq == 1000000000:
            #if (i + 1) % self.check_freq == 0:
                print(f"Current self-play batch: {i+1}", file=sys.stderr)
                win_ratio = self.policy_evaluate()
                writer.add_scalar('WinRatio', win_ratio, i)
                self.policy_value_net.save_model('./current_policy.model')
                if win_ratio > self.best_win_ratio:
                    print("New best policy!!!", file=sys.stderr)
                    self.best_win_ratio = win_ratio
                    self.policy_value_net.save_model('./best_policy.model')
                if win_ratio == 1.0:
                    print(f"Early stop!The hybrid model now can surely beat pure MCTS!", file=sys.stderr)
                    break
        writer.close()

if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
