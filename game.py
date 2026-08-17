#! /usr/bin/python3

import sys
import numpy as np

class Game:
    # The board is indexed [x][y] (consistent with Move=(x,y) on the
    # player/C++ side); width=board_width, height=board_height.
    def __init__(self, board_width=11, board_height=11):
        self.board_width = board_width
        self.board_height = board_height
        self.board = np.zeros((board_width, board_height), dtype=np.int32)

    # Returns (winner, steps).
    # NOTE(junhaozhang): the board must be cleared at game start, otherwise
    # consecutive evaluation games would be played on the previous game's
    # final position. Each player's own search tree must be reset() by the
    # caller before every game.
    def start_play(self, black_player, white_player):
        self.board = np.zeros((self.board_width, self.board_height), dtype=np.int32)
        players = [ black_player, white_player ]
        last_move = (-1, -1)
        current_player = 0 # black always moves first
        steps = 0
        while True:
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board, last_move, return_prob=False)
            # NOTE(junhaozhang): both players' search trees must apply this
            # move, otherwise StateEquals in the opponent's next get_action
            # will mismatch (it only recognizes the moves it has seen).
            for player in players:
                player.play(move)
            last_move = move
            self.board[move[0]][move[1]] = 2 * (1 - current_player) - 1
            steps += 1
            end, win = player_in_turn.check_end_win()
            if end:
                winner = current_player if win else -1
                return winner, steps
            current_player = 1 - current_player

    # NOTE(junhaozhang): the first temp_moves plies sample from visit counts
    # with temperature=1.0 for opening diversity; afterwards switch to tau->0
    # (play the strongest move), so the game outcome is decided by playing
    # strength rather than random moves -- this keeps the value target z clean
    # (AlphaGo Zero uses the first 30 plies on 19x19).
    # Note: in either phase, the mcts_probs stored as training data are always
    # the MCTS visit-count distribution at tau=1 -- that is the supervision
    # signal for the policy head; replacing it with one-hot would corrupt the
    # policy head training.
    def start_self_play(self, player, temperature=1.0, temp_moves=10):
        states, mcts_probs, current_players = [], [], [] 
        last_move = (-1, -1)
        current_player = 0
        steps = 0
        while True:
            states.append(self.__get_board_input_tensor(last_move, current_player))
	    #print(f"Geting action, last_move={last_move}", file=sys.stderr)
            temp = temperature if steps < temp_moves else 1e-3
            move, move_probs = player.get_action(self.board, last_move, temp, True, True)
	    #print(f"Playing postion {move}", file=sys.stderr)
            player.play(move)
            self.board[move[0]][move[1]] = 2 * (1 - current_player) - 1
	    #print(f"Board:\n{self.board}", file=sys.stderr)
            last_move = move
            mcts_probs.append(move_probs)
            current_players.append(current_player)
            steps += 1
            end, win = player.check_end_win()
            if end:
                winner = current_player if win else -1
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
 		#print(f"Steps: {steps}, winner={winner}", file=sys.stderr)
                self.board = np.zeros((self.board_width, self.board_height), dtype=np.int32)
                player.reset()
                return winner, zip(states, mcts_probs, winners_z)
            current_player = 1 - current_player

    # current_player: 0 = black, 1 = white
    # NOTE(junhaozhang): the planes must be indexed [y][x], consistent with
    # the C++ GenModelInputTensor and the policy flattening idx = y * W + x.
    # self.board is indexed [x][y], hence the transpose.
    #   channel 0: current player's stones
    #   channel 1: opponent's stones
    #   channel 2: last move position
    #   channel 3: side to move (all 1.0 if black is next)
    def __get_board_input_tensor(self, last_move, current_player):
        state = np.zeros((4, self.board_height, self.board_width))
        me = 1 if current_player == 0 else -1  # stone encoding: black +1, white -1
        state[0] = (self.board == me).T.astype(np.float64)
        state[1] = (self.board == -me).T.astype(np.float64)
        if last_move[0] >= 0 and last_move[1] >= 0:
            state[2][last_move[1]][last_move[0]] = 1.0
        if current_player == 0:  # black to move
            state[3][:,:] = 1.0
        return state
