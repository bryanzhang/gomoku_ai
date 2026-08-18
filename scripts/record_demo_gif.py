#! /usr/bin/python3
# Record one self-play game of a trained model and render it as an animated
# GIF (used to produce assets/selfplay_15x15.gif for the README).
#
# Usage:
#   ./scripts/record_demo_gif.py [model] [board_size] [sims] [out.gif]
#
# Example:
#   ./scripts/record_demo_gif.py model_examples/15x15_snapshots/policy_game_9000.model 15 400 assets/selfplay_15x15.gif
import os
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
import json
import sys

import torch  # must be imported before gomoku_ai (see the note in elo.py)
from game import Game
from player import AlphaZeroPlayer
from elo import prepare_model_path

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'model_examples/15x15_snapshots/policy_game_9000.model'
BOARD = int(sys.argv[2]) if len(sys.argv) > 2 else 15
SIMS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
OUT = sys.argv[4] if len(sys.argv) > 4 else 'assets/selfplay_15x15.gif'
CORES = 8


def record_game():
    ts_path = prepare_model_path(MODEL, 'gif', BOARD, BOARD)
    game = Game(BOARD, BOARD)
    players = [AlphaZeroPlayer(BOARD, SIMS, ts_path, CORES, 5.0, True),
               AlphaZeroPlayer(BOARD, SIMS, ts_path, CORES, 5.0, True)]
    last_move = (-1, -1)
    current = 0  # black first
    moves = []
    while True:
        player = players[current]
        move = player.get_action(game.board, last_move, temperature=1e-3, return_prob=False)
        for p in players:
            p.play(move)
        game.board[move[0]][move[1]] = 2 * (1 - current) - 1
        moves.append([int(move[0]), int(move[1])])
        last_move = move
        end, win = player.check_end_win()
        if end:
            winner = current if win else -1
            break
        current = 1 - current
    print(f'game over: winner={winner} (0=black,1=white,-1=draw), steps={len(moves)}', file=sys.stderr)
    return winner, moves


def render_gif(winner, moves):
    from PIL import Image, ImageDraw

    size = 600
    margin = 30
    cell = (size - 2 * margin) / (BOARD - 1)
    stars = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]

    def stone(d, px, py, black):
        r = cell / 2 - 2
        fill, outline = ((25, 25, 25), (0, 0, 0)) if black else ((245, 245, 245), (120, 120, 120))
        d.ellipse([px - r, py - r, px + r, py + r], fill=fill, outline=outline, width=2)
        hx, hy = px - r * 0.35, py - r * 0.35
        hr = r * 0.28
        d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(90, 90, 90) if black else (255, 255, 255))

    def base_board():
        img = Image.new('RGB', (size, size), (222, 176, 110))
        d = ImageDraw.Draw(img)
        for i in range(BOARD):
            p = margin + i * cell
            d.line([margin, p, size - margin, p], fill=(60, 40, 20), width=2)
            d.line([p, margin, p, size - margin], fill=(60, 40, 20), width=2)
        for sx, sy in stars:
            px, py = margin + sx * cell, margin + sy * cell
            d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(60, 40, 20))
        return img

    def find_win_line(black):
        color = 1 if black else -1
        grid = {(x, y): (1 if i % 2 == 0 else -1) for i, (x, y) in enumerate(moves)}
        for (x, y), v in grid.items():
            if v != color:
                continue
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                line = [(x + k * dx, y + k * dy) for k in range(5)]
                if all(grid.get(p) == color for p in line):
                    return line
        return None

    frames, durations = [], []
    for i in range(len(moves) + 1):
        img = base_board()
        d = ImageDraw.Draw(img)
        for j, (x, y) in enumerate(moves[:i]):
            stone(d, margin + x * cell, margin + y * cell, j % 2 == 0)
        if i == len(moves) and winner != -1:
            line = find_win_line(winner == 0)
            if line:
                p0 = (margin + line[0][0] * cell, margin + line[0][1] * cell)
                p1 = (margin + line[-1][0] * cell, margin + line[-1][1] * cell)
                d.line([p0, p1], fill=(220, 30, 30), width=6)
            text = ('Black' if winner == 0 else 'White') + ' wins'
            d.rectangle([size / 2 - 90, 12, size / 2 + 90, 44], fill=(255, 255, 255), outline=(0, 0, 0))
            d.text((size / 2 - 78, 20), text, fill=(200, 0, 0))
        frames.append(img)
        durations.append(500)
    durations[-1] = 4000  # hold the final frame
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    print(f'saved {OUT}: {len(frames)} frames', file=sys.stderr)


if __name__ == '__main__':
    winner, moves = record_game()
    render_gif(winner, moves)
