#! /usr/bin/python3

# 从 tensorboard event 文件绘制训练曲线, 与 train.py 写入的 tag 对应:
#   左图: Loss/Train + Entropy/Train(每局自对弈后的训练更新)
#   右图: Eval/OverallScore + Eval/BlackScore + Eval/WhiteScore(每隔 eval_freq 局评估)
# 断点续训会产生 step 重叠的 event 文件, 同一 step 保留最后写入的值。
#
# 用法:
#   ./plot_train_curve.py                                  # 默认读 ./gomoku_experiments, 输出 ./train_curve.png
#   ./plot_train_curve.py --dir ./gomoku_experiments --smooth 50 -o curve.png

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # 无显示环境
import matplotlib.pyplot as plt
import numpy as np

# (tag, 显示名, 颜色)
LOSS_TAGS = [('Loss/Train', 'loss', 'tab:red')]
ENTROPY_TAGS = [('Entropy/Train', 'entropy', 'tab:orange')]
EVAL_TAGS = [
    ('Eval/OverallScore', 'overall win ratio', 'tab:blue'),
    ('Eval/BlackScore', 'as black win ratio', 'tab:green'),
    ('Eval/WhiteScore', 'as white win ratio', 'tab:purple'),
]


# 读取 dir 下所有 event 文件里指定 tag 的 (step, value) 序列
def load_scalars(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logdir)
    ea.Reload()
    available = ea.Tags().get('scalars', [])
    series = {}
    for tag in available:
        events = ea.Scalars(tag)
        # 续训重叠 step 去重: 按写入时间保留最后一个
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
    parser.add_argument('--dir', default='./gomoku_experiments', help='tensorboard event 目录')
    parser.add_argument('-o', '--output', default='./train_curve.png', help='输出图片路径')
    parser.add_argument('--smooth', type=int, default=20,
                        help='滑动平均窗口(局数), 0 表示只画原始值')
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f'event 目录不存在: {args.dir}')
    series = load_scalars(args.dir)
    if not series:
        sys.exit(f'{args.dir} 里没有 scalar 数据')

    fig, (ax_loss, ax_eval) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: loss / entropy(双 y 轴, 合并图例)
    ax_entropy = ax_loss.twinx()
    plot_series(ax_loss, series, LOSS_TAGS, args.smooth, 'loss', legend=False)
    plot_series(ax_entropy, series, ENTROPY_TAGS, args.smooth, 'entropy', legend=False)
    handles = [ax_loss.get_legend_handles_labels(), ax_entropy.get_legend_handles_labels()]
    if any(h[0] for h in handles):
        ax_loss.legend(sum((h[0] for h in handles), []),
                       sum((h[1] for h in handles), []), loc='upper right', fontsize=9)
    ax_loss.set_xlabel('batch (self-play games)')
    ax_loss.set_title('Training loss / entropy')

    # 右图: 评估胜率
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
