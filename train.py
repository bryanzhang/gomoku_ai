#! /usr/bin/python3

# Training uses the v2 (ResNet) network; v1 (3conv) is kept in
# policy_value_net_pytorch.py for loading legacy weights.
import os
# Must be set before importing torch (pulled in indirectly by
# policy_value_net_pytorch_v2): NNPACK init fails on this machine and c10
# prints one WARNING per conv op; raise the level to ERROR to silence it.
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

# Default MCTS worker threads: one per local CPU core.
DEFAULT_CORES = os.cpu_count() or 1

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
        # Run one evaluation every this many self-play games
        # (with play_batch_size=1, one batch == one game).
        self.eval_freq = eval_freq
        self.eval_n_games = eval_games  # games per evaluation; an even number is recommended so black/white are split evenly
        # Self-play explores with temperature=1.0 for the first temp_moves
        # plies, then switches to tau->0 (play the strongest move).
        self.temp_moves = temp_moves
        self.save_freq = 10  # dump a checkpoint every this many batches
        self.n_playout = n_playout
        self.kl_targ = 0.02
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model,
                                               num_blocks=num_blocks, channels=channels)
        self.game = Game(self.board_width, self.board_height)
        # All on-disk/temp paths carry a board-size suffix (11x11 keeps the
        # historical paths unchanged), so training runs of different board
        # sizes never overwrite each other's checkpoints or race on the same
        # torchscript temp file.
        suffix = "" if board_size == 11 else f"_{board_size}x{board_size}"
        self.tmp_model_path = f"/tmp/gomoku_model{suffix}.pt"
        self.ckpt_path = f"./current_policy{suffix}.ckpt"
        self.model_path = f"./current_policy{suffix}.model"
        self.log_dir = f"./gomoku_experiments{suffix}"
        # Evaluation (similar to elo.py): play against a fixed opponent with
        # alternating colors, recording metrics only (overall/black/white
        # score, avg steps); no early stopping or model selection. The
        # opponent is model-free pure MCTS by default; when opp_model is
        # non-empty (.model/.ckpt/.pt) it is loaded as an AlphaZero opponent.
        self.eval_cur_ts_path = f"/tmp/gomoku_model_eval_cur{suffix}.pt"  # torchscript of the current net
        self.opp_model = opp_model
        self.opp_simulations = opp_simulations
        self.opp_c_puct = opp_c_puct
        self.opp_reuse_states = opp_reuse_states
        self.eval_opp_player = None  # built on first evaluation, then reused (opponent does not change during training)
        # Save a snapshot of the model at every evaluation, so any of them
        # can be picked for testing after training finishes.
        self.eval_snapshot_dir = f"./{board_size}x{board_size}_snapshots"
        self.mcts_player = AlphaZeroPlayer(board_size, self.n_playout, self.tmp_model_path, DEFAULT_CORES, 5.0, True)
        # Replay buffer: with 8x augmentation each game yields ~400 samples,
        # so 10000 entries hold only ~25 recent games of experience.
        self.data_buffer = deque(maxlen=10000)
        self.epochs = 5 # num of train_steps for each update
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0 # adaptively adjust the learning rate based on KL
        self.start_batch = 0
        # Resume training progress from the checkpoint (the elo/best_elo
        # fields in old checkpoints are deprecated and ignored).
        extra = self.policy_value_net.extra_state
        if extra:
            self.start_batch = extra.get('batch', 0)
            self.lr_multiplier = extra.get('lr_multiplier', 1.0)
            print(f"Resumed from batch {self.start_batch}, lr_multiplier={self.lr_multiplier:.3f}", file=sys.stderr)

    def get_augumented_data(self, play_data):
        # Augment samples by rotation and flipping, producing 8x more data.
        # TODO(junhaozhang): half of the samples could additionally be
        # color-swapped (black <-> white) for another 4x.
        # NOTE(junhaozhang): state planes are indexed [y][x] and mcts_prob is
        # flattened with idx = y*W + x; both share the same layout, so applying
        # exactly the same geometric transform to state and prob map is
        # correct -- no extra flipud needed.
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

    # Build the evaluation opponent. Pure MCTS needs no model export; for an
    # AlphaZero opponent, non-.pt models (.model/.ckpt) are first exported to
    # torchscript with the v1/v2 architecture auto-detected from the content.
    def build_eval_opponent(self):
        if not self.opp_model:
            desc = (f"PureMCTS(sims={self.opp_simulations}, c_puct={self.opp_c_puct}, "
                    f"reuse_states={self.opp_reuse_states})")
            return PureMCTSPlayer(self.board_size, self.opp_simulations, DEFAULT_CORES, self.opp_c_puct,
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
        return AlphaZeroPlayer(self.board_size, self.opp_simulations, ts_path, DEFAULT_CORES, self.opp_c_puct,
                               self.opp_reuse_states), desc

    # Fixed-opponent evaluation (similar to elo.py): the current model and
    # the opponent alternate playing black, each taking black for half of the
    # games. Returns (overall_score, black_score, white_score, avg_steps),
    # all from the current model's perspective.
    def policy_evaluate(self, games_played):
        start_time = time.time()
        # NOTE(junhaozhang): a dedicated Game instance is required!
        # start_self_play relies on self.game's board being empty at the end
        # of the previous game; if evaluation reused self.game, the final
        # position of an eval game would linger on the board and break
        # StateEquals on the first move of the next self-play game.
        eval_game = Game(self.board_width, self.board_height)
        # Snapshot the model (state_dict) exactly as seen by this evaluation,
        # so it can be picked for testing afterwards.
        os.makedirs(self.eval_snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(self.eval_snapshot_dir, f'policy_game_{games_played}.model')
        self.policy_value_net.save_model(snapshot_path)
        # The current model must be re-exported before every evaluation
        # (its weights keep changing during training).
        self.policy_value_net.save_model_with_torchscript(self.eval_cur_ts_path)
        current_player = AlphaZeroPlayer(self.board_size, self.n_playout, self.eval_cur_ts_path, DEFAULT_CORES, 5.0, True)
        if self.eval_opp_player is None:
            self.eval_opp_player, self.eval_opp_desc = self.build_eval_opponent()
            print(f"[Eval] opponent: {self.eval_opp_desc}", file=sys.stderr)
        opponent = self.eval_opp_player

        n = self.eval_n_games
        stats = {'black': [0, 0, 0], 'white': [0, 0, 0]}  # current model's view: [win, lose, draw]
        total_steps = 0
        for i in range(n):
            # Reset both players' search trees (keeping thread pools) before
            # each game, otherwise the board state won't match the next game.
            current_player.reset()
            opponent.reset()
            if i % 2 == 0:
                winner, steps = eval_game.start_play(current_player, opponent)
                key = 'black'
            else:
                winner, steps = eval_game.start_play(opponent, current_player)
                if winner != -1:
                    winner = 1 - winner  # map back to current model's view: 0=win, 1=lose, -1=draw
                key = 'white'
            stats[key][0 if winner == 0 else (1 if winner == 1 else 2)] += 1
            total_steps += steps

        wins = stats['black'][0] + stats['white'][0]
        losses = stats['black'][1] + stats['white'][1]
        draws = stats['black'][2] + stats['white'][2]
        overall_score = (wins + 0.5 * draws) / n
        # With odd n the black/white game counts differ; normalize separately.
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
                # Fixed-opponent evaluation; metrics go to tensorboard only
                # (compare with the train curve), no early stopping.
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
                        help='initial model (.model/.ckpt); train from scratch if omitted')
    parser.add_argument('--board-size', type=int, default=11, choices=[8, 11, 15],
                        help='board edge length (8/11/15, matching the gomoku_ai C++ bindings); '
                             'checkpoints/snapshots/tensorboard dirs of 15x15 runs automatically '
                             'get a _15x15 suffix and will not overwrite 11x11 artifacts')
    parser.add_argument('--num-blocks', type=int, default=3,
                        help='number of residual blocks of the policy-value net (v2 ResNet); '
                             'consider a larger value for bigger boards')
    parser.add_argument('--channels', type=int, default=64,
                        help='number of channels of the policy-value net (v2 ResNet); '
                             'consider a larger value for bigger boards')
    parser.add_argument('--n-playout', type=int, default=1000,
                        help='MCTS simulations per move for the current model, both in '
                             'self-play and evaluation')
    parser.add_argument('--temp-moves', type=int, default=15,
                        help='explore with temperature=1.0 for the first N plies of self-play, '
                             'then switch to tau->0 (strongest move)')
    parser.add_argument('--games', type=int, default=10000, help='total number of self-play games')
    parser.add_argument('--eval-freq', type=int, default=1000, help='evaluate every N games')
    parser.add_argument('--eval-games', type=int, default=20,
                        help='number of games per evaluation; an even number is recommended '
                             'so black/white are split evenly')
    parser.add_argument('--opp-model', default='',
                        help='evaluation opponent model (.model/.ckpt/.pt, v1/v2 auto-detected); '
                             'empty means model-free pure MCTS')
    parser.add_argument('--opp-simulations', type=int, default=500000,
                        help='MCTS simulations per move of the evaluation opponent')
    parser.add_argument('--opp-c-puct', type=float, default=2.0, help='PUCT constant of the evaluation opponent')
    parser.add_argument('--opp-reuse-states', action=argparse.BooleanOptionalAction, default=True,
                        help='whether the evaluation opponent reuses its search tree')
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
