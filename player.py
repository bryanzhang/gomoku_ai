#! /usr/bin/python3

from abc import ABC, abstractmethod
import gomoku_ai
import numpy as np
import sys

# Pick the C++ binding class by board size (e.g. board_size=15 ->
# PureMCTSFramework15 / AlphaZeroMCTSFramework15)
def _framework_class(prefix, board_size):
    cls = getattr(gomoku_ai, f'{prefix}{board_size}', None)
    if cls is None:
        raise ValueError(f'gomoku_ai does not support {board_size}x{board_size} boards (no {prefix}{board_size} binding)')
    return cls

class PureMCTSPlayer:
    def __init__(self, board_size, simulate_times, cores, c_puct, reuse_states):
        self.board_size = board_size
        self.framework_cls = _framework_class('PureMCTSFramework', board_size)
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

        # NOTE(junhaozhang): the C++ side state includes every stone played
        # (constructed with the full board including last_move, then Play for
        # each move), same as AlphaZeroPlayer. StateEquals must therefore be
        # checked against the full board -- do NOT remove the last_move stone
        # (removing it would both mismatch and corrupt Game's board reference).
        if not self.game:
            self.game = self.framework_cls(self.cores, np_board, last_move, self.c_puct, self.reuse_states)
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
        # If the framework exists, only clear the search tree but keep the
        # thread pool (C++ Reset), avoiding rebuilding the thread pool every
        # game; if game is None, keep lazy construction (the first get_action
        # builds from the board at that moment, supporting joining mid-game).
        if self.game:
            self.game.Reset()

    def play(self, move):
        # game may not exist yet before this player's first move; the first
        # get_action will construct it from the current board directly.
        if not self.game:
            return False, False
        self.game.Play(move[0], move[1])
        if not self.game.IsEnd():
            return False, False
        return True, self.game.AvailableCount() > 0

class AlphaZeroPlayer:
    # Temperatures below this threshold are treated as tau->0: play the most
    # visited move directly.
    GREEDY_TEMP = 1e-2

    def __init__(self, board_size, simulate_times, model_path, cores, c_puct, reuse_states):
        self.board_size = board_size
        self.cores = cores
        self.c_puct = c_puct
        self.reuse_states = reuse_states
        self.game = _framework_class('AlphaZeroMCTSFramework', board_size)(cores, c_puct, reuse_states)
        self.simulate_times = simulate_times
        self.model_path = model_path

    def check_end_win(self):
        if self.game.AvailableCount() == 0:
            return True, False
        if self.game.IsEnd():
            return True, True
        return False, False

    # Reshape the tau=1 visit-count distribution into a move-selection
    # distribution by temperature: p' ∝ p^(1/tau).
    # Degenerates to argmax (random among ties) as tau->0, avoiding the
    # underflow of exp(log(p)/1e-3).
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

    # NOTE(junhaozhang): MCTS always returns the normalized visit-count
    # distribution at temperature=1.0, which is the training target of the
    # policy head (same as AlphaZero -- what is stored is always N/sum(N));
    # the `temperature` argument only decides how THIS move is actually
    # picked and never pollutes the training target.
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
            raise RuntimeError('State not equal!')

        move_probs = np.zeros(self.board_size * self.board_size)
        sensible_moves, sensible_probs = self.game.SearchBestMove(self.simulate_times, self.model_path, 1.0)
        move_probs[sensible_moves] = sensible_probs  # training target: always the tau=1 visit-count distribution
        visit_probs = np.array(sensible_probs)
        select_probs = self.selection_probs(visit_probs, temperature)
        # Inject Dirichlet noise only during the exploration phase (large
        # tau). Adding 25% noise in the greedy tau->0 phase would
        # deliberately play a wasted move every few plies, turning the value
        # target z into a noisy label.
        if self_play and temperature > self.GREEDY_TEMP:
            select_probs = 0.75 * select_probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(select_probs)))
        move = np.random.choice(sensible_moves, p=select_probs)
        move = (move % self.board_size, move // self.board_size)
        if not return_prob:
            return move
        return move, move_probs

    def reset(self):
        # NOTE(junhaozhang): do NOT create a new Framework per game -- a new
        # thread pool yields new thread ids, and the C++ ThreadLocalModels
        # would permanently accumulate torch modules keyed by thread id
        # (+cores copies per game, OOM after a few games). Reset only clears
        # the search tree; the thread pool and model cache are reused across
        # games.
        self.game.Reset()

    def play(self, move):
        self.game.Play(move[0], move[1])
        if not self.game.IsEnd():
            return False, False
        return True, self.game.AvailableCount() > 0
