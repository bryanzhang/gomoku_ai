#! /usr/bin/python3

from abc import ABC, abstractmethod
import gomoku_ai
import numpy as np

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
            print(f'State not equal!')
            raise

        return self.game.SearchBestMove(self.simulate_times)

    def play(self, move):
        self.game.Play(move[0], move[1])
        if not self.game.IsEnd():
            return False, False
        return True, self.game.AvailableCount() > 0

class AlphaZeroPlayer:
    def __init__(self, simulate_times, model_path):
        self.game = gomoku_ai.AlphaZeroMCTSFramework11(1, 5.0, False)
        self.simulate_times = simulate_times
        self.model_path = model_path

    def get_action(self, np_board, last_move, temperature=1e-3, return_prob=True):
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
            print('State not equal!')
            raise

        move_probs = np.zeros(11 * 11)
        sensible_moves = []
        sensible_probs = []
        self.game.SearchBestMove(self.simulate_times, self.model_path, temperature, sensible_moves, sensible_probs)
        move_probs[moves] = probs
        if self.is_selfplay:
            move = np.random.choice(sensible_moves, p=0.75*sensible_probs + 0.25 * np.random.dirichlet(0.3*np.ones(len(probs))))
        else:
            move = np.random.choice(sensible_moves, p=sensible_probs)
        if not return_prob:
            return move
        return move, move_probs

    def play(self, move):
        self.game.Play(move[0], move[1])
        if not self.game.IsEnd():
            return False, False
        return True, self.game.AvailableCount() > 0
