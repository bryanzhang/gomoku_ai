#! /usr/bin/python3

from abc import ABC, abstractmethod
import gomoku_ai
import numpy as np
import sys

class PureMCTSPlayer:
    def __init__(self, simulate_times, cores, c_puct, reuse_states):
        self.cores = cores
        self.c_puct = c_puct
        self.simulate_times = simulate_times
        self.reuse_states = reuse_states
        self.game = None

    def get_action(self, np_board, last_move, return_prob=False):
        x = last_move[0]
        y = last_move[1]
        if x < 0 or y < 0:
            is_last_black = False
        else:
            is_last_black = (np_board[x][y] == 1)

        # NOTE(junhaozhang): C++ 侧状态包含所有已落子(构造按含 last_move 的完整棋盘,
        # Play 落每一手), 与 AlphaZeroPlayer 一致, 这里要拿完整棋盘做 StateEquals,
        # 不能把 last_move 的子摘掉(摘掉既会失配, 也会腐蚀 Game 的 board 引用)。
        if not self.game:
            self.game = gomoku_ai.PureMCTSFramework11(self.cores, np_board, last_move, self.c_puct, self.reuse_states)
        elif not self.game.StateEquals(np_board, is_last_black):
            print(f'State not equal!', file=sys.stderr)
            raise RuntimeError('State not equal!')

        move = self.game.SearchBestMove(self.simulate_times)
        if return_prob:
            return move, 1.0
        else:
            return move

    def check_end_win(self):
        if self.game.AvailableCount() == 0:
            return True, False
        if self.game.IsEnd():
            return True, True
        return False, False

    def reset(self):
        # 有 framework 时只清搜索树、保留线程池(C++ Reset), 避免每局重建 16 个线程;
        # game 为 None 时保持惰性构造(首次 get_action 按当时棋盘建, 支持中途入局)。
        if self.game:
            self.game.Reset()

    def play(self, move):
        # 还没轮到自己走过棋时 game 尚未创建, 首次 get_action 会直接按当前棋盘构建
        if not self.game:
            return False, False
        self.game.Play(move[0], move[1])
        if not self.game.IsEnd():
            return False, False
        return True, self.game.AvailableCount() > 0

class AlphaZeroPlayer:
    # temperature 小于该阈值就当作 τ→0 处理, 直接走访问次数最多的一手
    GREEDY_TEMP = 1e-2

    def __init__(self, simulate_times, model_path, cores, c_puct, reuse_states):
        self.cores = cores
        self.c_puct = c_puct
        self.reuse_states = reuse_states
        self.game = gomoku_ai.AlphaZeroMCTSFramework11(cores, c_puct, reuse_states)
        self.simulate_times = simulate_times
        self.model_path = model_path

    def check_end_win(self):
        if self.game.AvailableCount() == 0:
            return True, False
        if self.game.IsEnd():
            return True, True
        return False, False

    # 把 τ=1 的访问次数分布按温度重整成"选点分布": p'∝ p^(1/τ)
    # τ→0 时退化成 argmax(并列随机), 避免 exp(log(p)/1e-3) 直接下溢
    @classmethod
    def selection_probs(cls, visit_probs, temperature):
        if temperature <= cls.GREEDY_TEMP:
            probs = np.zeros_like(visit_probs)
            best = np.flatnonzero(visit_probs == visit_probs.max())
            probs[np.random.choice(best)] = 1.0
            return probs
        if temperature == 1.0:
            return visit_probs
        logits = np.log(visit_probs + 1e-10) / temperature
        probs = np.exp(logits - logits.max())
        return probs / probs.sum()

    # NOTE(junhaozhang): MCTS 固定按 temperature=1.0 返回访问次数归一化分布, 它是
    # policy head 的训练目标(与 AlphaZero 一致, 存的永远是 N/ΣN); 入参 temperature
    # 只决定"这一手实际怎么选", 不会污染训练目标。
    def get_action(self, np_board, last_move, temperature=1e-3, return_prob=True, self_play=False):
        x = last_move[0]
        y = last_move[1]
        if x < 0 or y < 0:
            is_last_black = False
            last_piece = None
        else:
            is_last_black = (np_board[x][y] == 1)
            last_piece = np_board[x][y]
            #np_board[x][y] = 0

        if not self.game.StateEquals(np_board, is_last_black):
            print('State not equal!', file=sys.stderr)
            raise

        move_probs = np.zeros(11 * 11)
        sensible_moves, sensible_probs = self.game.SearchBestMove(self.simulate_times, self.model_path, 1.0)
        move_probs[sensible_moves] = sensible_probs  # 训练目标: 始终是 τ=1 的访问次数分布
        visit_probs = np.array(sensible_probs)
        select_probs = self.selection_probs(visit_probs, temperature)
        # 只在探索阶段(τ 较大)注入 Dirichlet 噪声。τ→0 的贪心阶段再加 25% 噪声等于
        # 隔几手就故意走一步废棋, 会把 value 目标 z 打成噪声标签。
        if self_play and temperature > self.GREEDY_TEMP:
            select_probs = 0.75 * select_probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(select_probs)))
        move = np.random.choice(sensible_moves, p=select_probs)
        move = (move % 11, move // 11)
        if not return_prob:
            return move
        return move, move_probs

    def reset(self):
        # NOTE(junhaozhang): 不能每局新建 Framework —— 新线程池产生新 thread id,
        # C++ ThreadLocalModels 会按 thread id 永久累积 torch 模块(每局 +cores 份,
        # 打几局就 OOM)。Reset 只清搜索树, 线程池与模型缓存跨局复用。
        self.game.Reset()

    def play(self, move):
        self.game.Play(move[0], move[1])
        if not self.game.IsEnd():
            return False, False
        return True, self.game.AvailableCount() > 0
