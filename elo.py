#! /usr/bin/python3

# 两个 player 的 Elo 跑分脚本: 交替执黑, 各执黑一半局数, 输出分先手的胜率和 Elo 差。
# 每个 player 都可以是 AlphaZero(带模型文件) 或纯 MCTS(model-free):
# 由 --{a,b}-model 是否非空决定 —— 非空用 AlphaZero, 置空(--a-model=)用纯 MCTS。
#
# 示例:
#   ./elo.py                                        # 默认: A=AlphaZero(./current_policy.model) vs B=纯MCTS(10万次模拟)
#   ./elo.py -n 40 --b-model ./best_policy.model    # 两个模型对打
#   ./elo.py --a-model ./gomoku_model.pt --a-simulations 2000
#   ./elo.py --a-model= --a-simulations 50000       # A 也用纯 MCTS

import argparse
import math
import os
import sys
import tempfile
import time

# 必须在 import torch(由下面 policy_value_net_pytorch_v2 间接引入)之前设置:
# 本机 NNPACK 初始化失败, c10 每次 conv 都打一条 WARNING 刷屏, 提到 ERROR 级屏蔽。
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')

# NOTE(junhaozhang): gomoku_ai.so 依赖 libtorch 动态库, 必须先 import torch
# (policy_value_net_pytorch_v2 会间接 import torch) 再 import player/game, 否则 .so 加载失败。
from policy_value_net_pytorch_v2 import load_net_any_arch
from game import Game
from player import AlphaZeroPlayer, PureMCTSPlayer

BOARD_W = BOARD_H = 11


# Game.start_play 调 get_action 时只传 return_prob, temperature 由这个子类注入,
# 避免改 Game 的通用接口。
class AlphaZeroPlayerWithTemp(AlphaZeroPlayer):
    def __init__(self, *args, temperature=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def get_action(self, np_board, last_move, return_prob=False):
        return super().get_action(np_board, last_move,
                                  temperature=self.temperature,
                                  return_prob=return_prob)


# AlphaZero 的 C++ 侧只认 torchscript(.pt); .model(state_dict/checkpoint) 先导出成 .pt。
# 权重结构(v1 3conv / v2 ResNet)由 load_net_any_arch 按内容自动识别, 新旧 .model 都能转。
def prepare_model_path(model_path, tag):
    if model_path.endswith('.pt'):
        return model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"player {tag}: model file not found: {model_path}")
    net = load_net_any_arch(BOARD_W, BOARD_H, model_path)
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
    # 模型路径非空 -> AlphaZero(带模型); 空 -> 纯 MCTS(model-free)
    if not model:
        player = PureMCTSPlayer(sims, cores, c_puct, reuse)
        desc = f"PureMCTS(sims={sims}, cores={cores}, c_puct={c_puct}, reuse_states={reuse})"
    else:
        temp = getattr(args, f'{tag}_temperature')
        ts_path = prepare_model_path(model, tag)
        player = AlphaZeroPlayerWithTemp(sims, ts_path, cores, c_puct, reuse, temperature=temp)
        desc = (f"AlphaZero(model={model}, sims={sims}, cores={cores}, c_puct={c_puct}, "
                f"reuse_states={reuse}, temperature={temp})")
    name = getattr(args, f'{tag}_name') or tag.upper()
    return player, name, desc


# 得分率 -> 相对 Elo 差: diff = 400 * log10(S / (1 - S))
# 全胜/全负时 log 发散, 用 1/(2*(n+1)) 截断, 相当于给两边各加半局先验。
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

    def add_player_args(tag, model, sims, c_puct):
        g = parser.add_argument_group(f'player {tag}')
        g.add_argument(f'--{tag}-model', default=model,
                       help=f'player {tag} model file: .model/.ckpt state_dict/checkpoint '
                            f'或 .pt torchscript(直接使用); 非空即 AlphaZero, 置空(--{tag}-model=)'
                            f'则为纯 MCTS(model-free)')
        g.add_argument(f'--{tag}-simulations', type=int, default=sims,
                       help=f'player {tag} MCTS simulation count per move')
        g.add_argument(f'--{tag}-cores', type=int, default=16,
                       help=f'player {tag} thread count')
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

    game = Game()
    # 按 A 的视角统计: [胜, 负, 和], 再按 A 执黑/执白分开
    stats = {'black': [0, 0, 0], 'white': [0, 0, 0]}
    start_time = time.time()
    try:
        for i in range(n_games):
            # 每局重建双方搜索树, 否则下一局棋盘状态对不上
            player_a.reset()
            player_b.reset()
            t0 = time.time()
            a_is_black = (i % 2 == 0)
            if a_is_black:
                winner, steps = game.start_play(player_a, player_b)
            else:
                winner, steps = game.start_play(player_b, player_a)
                if winner != -1:
                    winner = 1 - winner  # 换算回 A 的视角: 0=A胜, 1=A负, -1=和
            # winner: 0 -> A胜, 1 -> A负, -1 -> 和
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
    # 正态近似 95% 置信区间; 用截断后的得分率算标准误, 避免全胜/全负时退化成点
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
