#! /usr/bin/python3

# 模型循环联赛(round-robin): 每对模型对打若干局(执黑执白各一半),
# 结束输出总积分排名(赢+1, 和+0.5, 输+0)与两两对战得分矩阵。
#
# 示例:
#   ./league.py --models=a.model,b.model,c.model                  # 默认每对 10 局
#   ./league.py --models=a.model,./eval_snapshots/policy_game_5000.model \
#               --games=20 --simulations=2000 --c-puct=4.0

import argparse
import sys
import time
from collections import defaultdict

# 复用 elo.py 的模型装载逻辑: .pt 直接用, .model/.ckpt 自动识别 v1/v2 导出 torchscript
from elo import prepare_model_path
from game import Game
from player import AlphaZeroPlayer


# 用 basename 做显示名; 重名时追加序号
def make_labels(paths):
    import os
    labels = [os.path.basename(p) for p in paths]
    seen = defaultdict(int)
    out = []
    for i, label in enumerate(labels):
        seen[label] += 1
        out.append(label if seen[label] == 1 and labels.count(label) == 1
                   else f'{label}#{seen[label]}')
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Round-robin league between gomoku models (AlphaZero players).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--models', required=True,
                        help='逗号分隔的模型路径列表(.model/.ckpt/.pt)')
    parser.add_argument('--games', type=int, default=10,
                        help='每对模型的对局数(建议偶数, 执黑执白各一半)')
    parser.add_argument('--simulations', type=int, default=1000, help='每手 MCTS 模拟次数')
    parser.add_argument('--c-puct', type=float, default=5.0, help='PUCT 常数')
    parser.add_argument('--cores', type=int, default=16, help='每个模型的搜索线程数')
    parser.add_argument('--reuse-states', action=argparse.BooleanOptionalAction, default=True,
                        help='复用搜索树')
    args = parser.parse_args()

    model_paths = [p.strip() for p in args.models.split(',') if p.strip()]
    n = len(model_paths)
    if n < 2:
        parser.error('至少需要 2 个模型')
    if args.games <= 0:
        parser.error('--games 必须为正')
    if args.games % 2 != 0:
        print(f'Warning: 每对 {args.games} 局为奇数, 黑白分配不完全均衡。', file=sys.stderr)

    labels = make_labels(model_paths)
    players = []
    for label, path in zip(labels, model_paths):
        ts_path = prepare_model_path(path, f'league_{label}')
        players.append(AlphaZeroPlayer(args.simulations, ts_path, args.cores,
                                       args.c_puct, args.reuse_states))
    print(f'League: {n} models, {args.games} games per pair, '
          f'sims={args.simulations}, c_puct={args.c_puct}, cores={args.cores}, '
          f'reuse_states={args.reuse_states}', file=sys.stderr)
    for label, path in zip(labels, model_paths):
        print(f'  {label}: {path}', file=sys.stderr)

    game = Game()
    # records[i][j] = [胜, 负, 和] (i 的视角, i 对 j); matrix[i][j] = i 从 j 身上拿到的分
    records = [[[0, 0, 0] for _ in range(n)] for _ in range(n)]
    matrix = [[0.0] * n for _ in range(n)]
    start_time = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            t0 = time.time()
            for g in range(args.games):
                players[i].reset()
                players[j].reset()
                black, white = (i, j) if g % 2 == 0 else (j, i)
                winner, steps = game.start_play(players[black], players[white])
                # winner: 0=黑胜, 1=白胜, -1=和; 换算成分数
                if winner == -1:
                    matrix[i][j] += 0.5
                    matrix[j][i] += 0.5
                    records[i][j][2] += 1
                    records[j][i][2] += 1
                    result = 'draw'
                else:
                    w, l = (black, white) if winner == 0 else (white, black)
                    matrix[w][l] += 1.0
                    records[w][l][0] += 1
                    records[l][w][1] += 1
                    result = f'{labels[w]} win'
                print(f'[{labels[i]} vs {labels[j]}] game {g + 1}/{args.games}: '
                      f'black={labels[black]}, {result}, steps={steps}, '
                      f'{time.time() - t0:.1f}s elapsed', file=sys.stderr)
            print(f'[pair done] {labels[i]} vs {labels[j]}: '
                  f'{matrix[i][j]:g} : {matrix[j][i]:g} '
                  f'({time.time() - t0:.1f}s)', file=sys.stderr)

    total_time = time.time() - start_time
    total_scores = [sum(matrix[i][j] for j in range(n) if j != i) for i in range(n)]
    total_games = args.games * (n - 1)
    wld = [[sum(records[i][j][k] for j in range(n)) for k in range(3)] for i in range(n)]

    # ---- 积分排名表 ----
    ranking = sorted(range(n), key=lambda i: -total_scores[i])
    headers = ['Rank', 'Model', 'Score', 'Win', 'Lose', 'Draw', 'WinRate']
    rows = [[str(r + 1), labels[i], f'{total_scores[i]:g} / {total_games}',
             str(wld[i][0]), str(wld[i][1]), str(wld[i][2]),
             f'{total_scores[i] / total_games:.3f}']
            for r, i in enumerate(ranking)]
    print('\n===== League Ranking =====')
    print_table(headers, rows)
    print(f'Total games: {args.games * n * (n - 1) // 2}, '
          f'total time: {total_time:.1f}s ({total_time / 60:.1f}min)')

    # ---- 两两对战得分矩阵: 单元格 = 行模型从列模型身上拿到的分 ----
    print(f'\n===== Head-to-head scores (row vs column, {args.games} games per pair) =====')
    headers = [''] + labels
    rows = []
    for i in range(n):
        row = [labels[i]]
        for j in range(n):
            row.append('—' if i == j else f'{matrix[i][j]:g}')
        rows.append(row)
    print_table(headers, rows)


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for k, cell in enumerate(row):
            widths[k] = max(widths[k], len(cell))
    fmt = ' | '.join(f'{{:<{w}}}' for w in widths)
    print(fmt.format(*headers))
    print('-+-'.join('-' * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


if __name__ == '__main__':
    main()
