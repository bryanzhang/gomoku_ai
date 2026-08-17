#! /usr/bin/python3

# Plot training curves from tensorboard event files, matching the tags written
# by train.py:
#   left:  Loss/Train + Entropy/Train (training update after each self-play game)
#   right: Eval/OverallScore + Eval/BlackScore + Eval/WhiteScore (evaluated every eval_freq games)
# Resumed training produces event files with overlapping steps; for the same
# step the last written value wins.
#
# Usage:
#   ./plot_train_curve.py                                  # reads ./gomoku_experiments, writes ./train_curve.png
#   ./plot_train_curve.py --dir ./gomoku_experiments --smooth 50 -o curve.png

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # headless environment
import matplotlib.pyplot as plt
import numpy as np

# (tag, display name, color)
LOSS_TAGS = [('Loss/Train', 'loss', 'tab:red')]
ENTROPY_TAGS = [('Entropy/Train', 'entropy', 'tab:orange')]
EVAL_TAGS = [
    ('Eval/OverallScore', 'overall win ratio', 'tab:blue'),
    ('Eval/BlackScore', 'as black win ratio', 'tab:green'),
    ('Eval/WhiteScore', 'as white win ratio', 'tab:purple'),
]


# Read the (step, value) series of the given tags from all event files under dir.
def load_scalars(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logdir)
    ea.Reload()
    available = ea.Tags().get('scalars', [])
    series = {}
    for tag in available:
        events = ea.Scalars(tag)
        # De-duplicate overlapping steps from resumed runs: keep the last write.
        by_step = {}
        for e in sorted(events, key=lambda e: (e.step, e.wall_time)):
            by_step[e.step] = e.value
        steps = sorted(by_step)
        series[tag] = (np.array(steps), np.array([by_step[s] for s in steps]))
    return series


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_series(ax, series, tags, smooth, ylabel, legend=True):
    plotted = False
    for tag, label, color in tags:
        if tag not in series:
            continue
        steps, values = series[tag]
        ax.plot(steps, values, color=color, alpha=0.25, linewidth=0.8)
        if smooth > 1:
            smoothed = moving_average(values, smooth)
            ax.plot(steps[len(steps) - len(smoothed):], smoothed, color=color,
                    linewidth=1.5, label=f'{label} (MA{smooth})')
        else:
            ax.plot(steps, values, color=color, linewidth=1.5, label=label)
        plotted = True
    ax.set_ylabel(ylabel)
    if plotted and legend:
        ax.legend(loc='best', fontsize=9)
    elif not plotted:
        ax.text(0.5, 0.5, f'no data for {ylabel}', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.grid(alpha=0.3)
    return plotted


def main():
    parser = argparse.ArgumentParser(
        description='Plot training curves (loss/entropy + eval win ratios) from tensorboard events.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', default='./gomoku_experiments', help='tensorboard event directory')
    parser.add_argument('-o', '--output', default='./train_curve.png', help='output image path')
    parser.add_argument('--smooth', type=int, default=20,
                        help='moving average window (in games); 0 plots raw values only')
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f'event directory does not exist: {args.dir}')
    series = load_scalars(args.dir)
    if not series:
        sys.exit(f'no scalar data in {args.dir}')

    fig, (ax_loss, ax_eval) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: loss / entropy (dual y-axes, merged legend)
    ax_entropy = ax_loss.twinx()
    plot_series(ax_loss, series, LOSS_TAGS, args.smooth, 'loss', legend=False)
    plot_series(ax_entropy, series, ENTROPY_TAGS, args.smooth, 'entropy', legend=False)
    handles = [ax_loss.get_legend_handles_labels(), ax_entropy.get_legend_handles_labels()]
    if any(h[0] for h in handles):
        ax_loss.legend(sum((h[0] for h in handles), []),
                       sum((h[1] for h in handles), []), loc='upper right', fontsize=9)
    ax_loss.set_xlabel('batch (self-play games)')
    ax_loss.set_title('Training loss / entropy')

    # Right: evaluation win ratios
    plot_series(ax_eval, series, EVAL_TAGS, 1, 'win ratio')
    ax_eval.set_xlabel('batch (self-play games)')
    ax_eval.set_ylim(-0.05, 1.05)
    ax_eval.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_eval.set_title('Eval win ratio vs opponent')

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f'saved: {args.output}')
    for tag in series:
        print(f'  {tag}: {len(series[tag][0])} points, last step={series[tag][0][-1]}')


if __name__ == '__main__':
    main()
