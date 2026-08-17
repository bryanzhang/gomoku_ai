#! /usr/bin/python3

# Round-robin league between models: each pair plays several games (half as
# black, half as white). At the end, print the total-score ranking
# (win +1, draw +0.5, loss +0) and the pairwise score matrix.
#
# Examples:
#   ./league.py --models=a.model,b.model,c.model                  # 10 games per pair by default
#   ./league.py --models=a.model,./eval_snapshots/policy_game_5000.model \
#               --games=20 --simulations=2000 --c-puct=4.0

import argparse
import os
import sys
import time
from collections import defaultdict

# Reuse elo.py's model loading: .pt is used directly, .model/.ckpt are
# auto-detected as v1/v2 and exported to torchscript.
from elo import prepare_model_path
from game import Game
from player import AlphaZeroPlayer

# Default MCTS worker threads: one per local CPU core.
DEFAULT_CORES = os.cpu_count() or 1


# Use the basename as display name; append a sequence number on collision.
def make_labels(paths):
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
                        help='comma-separated list of model paths (.model/.ckpt/.pt)')
    parser.add_argument('--board-size', type=int, default=11, choices=[8, 11, 15],
                        help='board edge length (8/11/15, must match the models\' network input size)')
    parser.add_argument('--games', type=int, default=10,
                        help='games per model pair (an even number is recommended so black/white are split evenly)')
    parser.add_argument('--simulations', type=int, default=1000, help='MCTS simulations per move')
    parser.add_argument('--c-puct', type=float, default=5.0, help='PUCT constant')
    parser.add_argument('--cores', type=int, default=DEFAULT_CORES,
                        help='search threads per model (default: local CPU count)')
    parser.add_argument('--reuse-states', action=argparse.BooleanOptionalAction, default=True,
                        help='reuse the search tree across moves')
    args = parser.parse_args()

    model_paths = [p.strip() for p in args.models.split(',') if p.strip()]
    n = len(model_paths)
    if n < 2:
        parser.error('at least 2 models are required')
    if args.games <= 0:
        parser.error('--games must be positive')
    if args.games % 2 != 0:
        print(f'Warning: odd games per pair ({args.games}), black/white split is not balanced.', file=sys.stderr)

    labels = make_labels(model_paths)
    players = []
    for label, path in zip(labels, model_paths):
        ts_path = prepare_model_path(path, f'league_{label}', args.board_size, args.board_size)
        players.append(AlphaZeroPlayer(args.board_size, args.simulations, ts_path, args.cores,
                                       args.c_puct, args.reuse_states))
    print(f'League: {n} models, {args.games} games per pair, '
          f'sims={args.simulations}, c_puct={args.c_puct}, cores={args.cores}, '
          f'reuse_states={args.reuse_states}', file=sys.stderr)
    for label, path in zip(labels, model_paths):
        print(f'  {label}: {path}', file=sys.stderr)

    game = Game(args.board_size, args.board_size)
    # records[i][j] = [win, lose, draw] (from i's perspective, i vs j);
    # matrix[i][j] = points i scored against j.
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
                # winner: 0=black wins, 1=white wins, -1=draw; map to points
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

    # ---- Score ranking table ----
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

    # ---- Head-to-head score matrix: cell = points the row model scored against the column model ----
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
