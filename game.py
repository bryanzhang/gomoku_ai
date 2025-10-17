#! /usr/bin/python3

import numpy as np

class Game:
    def __init__(self):
        self.board = np.zeros((11, 11), dtype=np.int32)

    # 返回赢家和步数
    def start_play(self, black_player, white_player):
        players = [ black_player, white_player ]
        last_move = (-1, -1)
        current_player = 0 # 总是黑棋先下
        steps = 0
        while True:
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board, last_move)
            player.play(move)
            last_move = move
            self.board[move[0]][move[1]] = 2 * (1 - current_player) - 1
            steps += 1
            end, win = player_in_turn.check_end_win()
            if end:
                winner = current_player if win else -1
                return winner, steps
            current_player = 1 - current_player

    def start_self_play(self, player, temperature=1e-3):
        states, mcts_probs, current_players = [], [], [] 
        last_move = (-1, -1)
        current_player = 0
        while True:
            states.append(self.get_board_input_tensor())
            move, move_probs = player.get_action(self.board, last_move, temperature, True)
            player.play(move)
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
                player.reset()
                return winner, zip(states, mcts_probs, winners_z)
            current_player = 1 - current_player
