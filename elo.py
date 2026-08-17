#! /usr/bin/python3

# Elo rating match between two players: alternate colors so each side plays
# black for half of the games; report per-color win rates and the Elo diff.
# Each player can be AlphaZero (with a model file) or pure MCTS (model-free),
# decided by whether --{a,b}-model is non-empty -- non-empty means AlphaZero,
# empty (--a-model=) means pure MCTS.
#
# Examples:
#   ./elo.py                                        # default: A=AlphaZero(./current_policy.model) vs B=pure MCTS(100k sims)
#   ./elo.py -n 40 --b-model ./best_policy.model    # model vs model
#   ./elo.py --a-model ./gomoku_model.pt --a-simulations 2000
#   ./elo.py --a-model= --a-simulations 50000       # A is pure MCTS too

import argparse
import math
import os
import sys
import tempfile
import time

# Must be set before importing torch (pulled in indirectly by
# policy_value_net_pytorch_v2 below): NNPACK init fails on this machine and
# c10 prints one WARNING per conv op; raise the level to ERROR to silence it.
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')

# NOTE(junhaozhang): gomoku_ai.so depends on libtorch shared libraries, so
# torch must be imported (indirectly via policy_value_net_pytorch_v2) before
# player/game, otherwise the .so fails to load.
from policy_value_net_pytorch_v2 import load_net_any_arch
from game import Game
from player import AlphaZeroPlayer, PureMCTSPlayer

# Default MCTS worker threads: one per local CPU core.
DEFAULT_CORES = os.cpu_count() or 1


# Game.start_play only passes return_prob to get_action; this subclass injects
# the temperature so Game's generic interface stays untouched.
class AlphaZeroPlayerWithTemp(AlphaZeroPlayer):
    def __init__(self, *args, temperature=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def get_action(self, np_board, last_move, return_prob=False):
        return super().get_action(np_board, last_move,
                                  temperature=self.temperature,
                                  return_prob=return_prob)


# The AlphaZero C++ side only accepts torchscript (.pt); .model
# (state_dict/checkpoint) files are exported to .pt first. The weight
# architecture (v1 3conv / v2 ResNet) is auto-detected by load_net_any_arch
# from the content, so both old and new .model files can be converted.
def prepare_model_path(model_path, tag, board_width=11, board_height=11):
    if model_path.endswith('.pt'):
        return model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"player {tag}: model file not found: {model_path}")
    net = load_net_any_arch(board_width, board_height, model_path)
    fd, ts_path = tempfile.mkstemp(prefix=f'elo_{tag}_', suffix='.pt')
    os.close(fd)
    net.save_model_with_torchscript(ts_path)
    print(f"player {tag}: exported torchscript {model_path} -> {ts_path}", file=sys.stderr)
    return ts_path


def build_player(tag, args):
    sims = getattr(args, f'{tag}_simulations')
    cores = getattr(args, f'{tag}_cores')
    c_puct = getattr(args, f'{tag}_c_puct')
    reuse = getattr(args, f'{tag}_reuse_states')
    model = getattr(args, f'{tag}_model')
    board_size = args.board_size
    # Non-empty model path -> AlphaZero (with model); empty -> pure MCTS (model-free)
    if not model:
        player = PureMCTSPlayer(board_size, sims, cores, c_puct, reuse)
        desc = f"PureMCTS(sims={sims}, cores={cores}, c_puct={c_puct}, reuse_states={reuse})"
    else:
        temp = getattr(args, f'{tag}_temperature')
        ts_path = prepare_model_path(model, tag, board_size, board_size)
        player = AlphaZeroPlayerWithTemp(board_size, sims, ts_path, cores, c_puct, reuse, temperature=temp)
        desc = (f"AlphaZero(model={model}, sims={sims}, cores={cores}, c_puct={c_puct}, "
                f"reuse_states={reuse}, temperature={temp})")
    name = getattr(args, f'{tag}_name') or tag.upper()
    return player, name, desc


# Score rate -> relative Elo diff: diff = 400 * log10(S / (1 - S)).
# log diverges on a perfect/zero score, so clamp with 1/(2*(n+1)) --
# equivalent to adding a half-game prior to both sides.
def score_to_elo_diff(score, n_games):
    eps = 1.0 / (2.0 * (n_games + 1))
    s = min(max(score, eps), 1.0 - eps)
    return 400.0 * math.log10(s / (1.0 - s))


def fmt_record(wins, losses, draws, n):
    score = (wins + 0.5 * draws) / n if n else 0.0
    return f"win {wins} / lose {losses} / draw {draws}, score {score:.3f} ({wins}/{n} wins)"


def main():
    parser = argparse.ArgumentParser(
        description='Elo match between two gomoku players (alternate black, half games each).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--games', type=int, default=20,
                        help='total number of games (should be even so each player gets black half the time)')
    parser.add_argument('--board-size', type=int, default=11, choices=[8, 9, 11, 15],
                        help='board edge length (pure MCTS supports 8/9/11/15, AlphaZero supports 8/11/15)')

    def add_player_args(tag, model, sims, c_puct):
        g = parser.add_argument_group(f'player {tag}')
        g.add_argument(f'--{tag}-model', default=model,
                       help=f'player {tag} model file: .model/.ckpt state_dict/checkpoint '
                            f'or .pt torchscript (used directly); non-empty means AlphaZero, '
                            f'empty (--{tag}-model=) means pure MCTS (model-free)')
        g.add_argument(f'--{tag}-simulations', type=int, default=sims,
                       help=f'player {tag} MCTS simulation count per move')
        g.add_argument(f'--{tag}-cores', type=int, default=DEFAULT_CORES,
                       help=f'player {tag} thread count (default: local CPU count)')
        g.add_argument(f'--{tag}-c-puct', type=float, default=c_puct,
                       help=f'player {tag} exploration constant')
        g.add_argument(f'--{tag}-reuse-states', action=argparse.BooleanOptionalAction, default=True,
                       help=f'player {tag} reuse search tree across moves')
        g.add_argument(f'--{tag}-temperature', type=float, default=1e-3,
                       help=f'player {tag} get_action temperature (alphazero only; <=1e-2 means greedy)')
        g.add_argument(f'--{tag}-name', default=None, help=f'player {tag} display name')

    add_player_args('a', './current_policy.model', 1000, 5.0)
    add_player_args('b', None, 100000, 2.0)
    args = parser.parse_args()

    n_games = args.games
    if n_games <= 0:
        parser.error('--games must be positive')
    if n_games % 2 != 0:
        print(f"Warning: odd games({n_games}), black/white split is not balanced.", file=sys.stderr)

    player_a, name_a, desc_a = build_player('a', args)
    player_b, name_b, desc_b = build_player('b', args)
    print(f"A = {desc_a}\nB = {desc_b}", file=sys.stderr)

    game = Game(args.board_size, args.board_size)
    # Stats from A's perspective: [win, lose, draw], split by A as black/white.
    stats = {'black': [0, 0, 0], 'white': [0, 0, 0]}
    start_time = time.time()
    try:
        for i in range(n_games):
            # Reset both players' search trees each game, otherwise the board
            # state won't match the next game.
            player_a.reset()
            player_b.reset()
            t0 = time.time()
            a_is_black = (i % 2 == 0)
            if a_is_black:
                winner, steps = game.start_play(player_a, player_b)
            else:
                winner, steps = game.start_play(player_b, player_a)
                if winner != -1:
                    winner = 1 - winner  # map back to A's view: 0=A wins, 1=A loses, -1=draw
            # winner: 0 -> A wins, 1 -> A loses, -1 -> draw
            rec = stats['black' if a_is_black else 'white']
            rec[0 if winner == 0 else (1 if winner == 1 else 2)] += 1
            result = {0: f'{name_a} win', 1: f'{name_b} win', -1: 'draw'}[winner]
            print(f"game {i + 1}/{n_games}: black={name_a if a_is_black else name_b}, "
                  f"{result}, steps={steps}, {time.time() - t0:.1f}s", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted, printing partial result...", file=sys.stderr)
        n_games = sum(stats['black']) + sum(stats['white'])
        if n_games == 0:
            return

    wins = stats['black'][0] + stats['white'][0]
    losses = stats['black'][1] + stats['white'][1]
    draws = stats['black'][2] + stats['white'][2]
    score = (wins + 0.5 * draws) / n_games
    elo_diff = score_to_elo_diff(score, n_games)
    # Normal-approximation 95% CI; compute the standard error from the clamped
    # score rate so a perfect/zero score does not degenerate to a point.
    eps = 1.0 / (2.0 * (n_games + 1))
    s_clamped = min(max(score, eps), 1.0 - eps)
    se = math.sqrt(s_clamped * (1.0 - s_clamped) / n_games)
    ci_low = score_to_elo_diff(score - 1.96 * se, n_games)
    ci_high = score_to_elo_diff(score + 1.96 * se, n_games)

    n_black = sum(stats['black'])
    n_white = sum(stats['white'])
    print(f"\n===== Elo Match Result ({n_games} games, {time.time() - start_time:.1f}s) =====")
    print(f"A: {desc_a}")
    print(f"B: {desc_b}")
    print(f"Overall   : {fmt_record(wins, losses, draws, n_games)}")
    print(f"As black  : {fmt_record(*stats['black'], n_black)}")
    print(f"As white  : {fmt_record(*stats['white'], n_white)}")
    print(f"Elo diff (A - B): {elo_diff:+.1f} (95% CI {ci_low:+.1f} ~ {ci_high:+.1f})")


if __name__ == '__main__':
    main()
