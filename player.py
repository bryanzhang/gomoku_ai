#! /usr/bin/python3

from abc import ABC, abstractmethod
import gomoku_ai
import numpy as np
import sys

class PureMCTSPlayer:
    def get_action(self, np_board, last_move):
        x = last_move[0]
        y = last_move[1]
        if x < 0 or y < 0:
            is_last_black = False
            last_piece = None
        else:
            is_last_black = (np_board[x][y] == 1)
            last_piece = np_board[x][y]
            np_board[x][y] = 0

        if not self.game:
            if last_piece != None:
                np_board[x][y] = last_piece
            self.game = gomoku_ai.GomokuMCTSFramework11(self.cores, np_board, last_move, self.c_puct, self.reuse_states)
        elif not self.game.StateEquals(np_board, is_last_black):
            print(f'State not equal!', file=sys.stderr)
            raise

        return self.game.SearchBestMove(self.simulate_times)

    def check_end_win(self):
        if self.game.AvailableCount() == 0:
            return True, False
        if self.game.IsEnd():
            return True, True
        return False, False

    def play(self, move):
        self.game.Play(move[0], move[1])
        if not self.game.IsEnd():
            return False, False
        return True, self.game.AvailableCount() > 0

class AlphaZeroPlayer:
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
        sensible_moves, sensible_probs = self.game.SearchBestMove(self.simulate_times, self.model_path, temperature)
        print(sensible_moves, file=sys.stderr)
        print(sensible_probs, file=sys.stderr)
        move_probs[sensible_moves] = sensible_probs
        sensible_probs = np.array(sensible_probs)
        if self_play:
            move = np.random.choice(sensible_moves, p=0.75*sensible_probs + 0.25 * np.random.dirichlet(0.3*np.ones(len(sensible_moves))))
        else:
            move = np.random.choice(sensible_moves, p=sensible_probs)
        move = (move % 11, move // 11)
        if not return_prob:
            return move
        return move, move_probs

    def reset(self):
        self.game = gomoku_ai.AlphaZeroMCTSFramework11(self.cores, self.c_puct, self.reuse_states)

    def play(self, move):
        self.game.Play(move[0], move[1])
        if not self.game.IsEnd():
            return False, False
        return True, self.game.AvailableCount() > 0
