#! /usr/bin/python3

import sys
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
            move = player_in_turn.get_action(self.board, last_move, return_prob=False)
            player_in_turn.play(move)
            last_move = move
            self.board[move[0]][move[1]] = 2 * (1 - current_player) - 1
            steps += 1
            end, win = player_in_turn.check_end_win()
            if end:
                winner = current_player if win else -1
                return winner, steps
            current_player = 1 - current_player

    # NOTE(junhaozhang): 自对弈必须用 temperature=1.0, 这样 MCTS 返回的是按访问次数
    # 归一化的分布(而非 argmax), 才能给 policy head 提供有效的监督信号。
    def start_self_play(self, player, temperature=1.0):
        states, mcts_probs, current_players = [], [], [] 
        last_move = (-1, -1)
        current_player = 0
        steps = 0
        while True:
            states.append(self.__get_board_input_tensor(last_move, current_player))
	    #print(f"Geting action, last_move={last_move}", file=sys.stderr)
            move, move_probs = player.get_action(self.board, last_move, temperature, True, True)
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
                self.board = np.zeros((11, 11), dtype=np.int32)
                player.reset()
                return winner, zip(states, mcts_probs, winners_z)
            current_player = 1 - current_player

    # current_player 0是黑棋，1是白棋
    # NOTE(junhaozhang): 平面必须按 [y][x] 索引, 与 C++ GenModelInputTensor 以及
    # policy 平铺下标 idx = y * 11 + x 保持一致。self.board 是 [x][y] 索引, 故需转置。
    #   通道0: 当前玩家的棋子位置
    #   通道1: 对手玩家的棋子位置
    #   通道2: 上一步落子位置
    #   通道3: 下一步谁下(黑棋为全1.0)
    def __get_board_input_tensor(self, last_move, current_player):
        state = np.zeros((4, 11, 11))
        me = 1 if current_player == 0 else -1  # 棋子编码: 黑棋+1, 白棋-1
        state[0] = (self.board == me).T.astype(np.float64)
        state[1] = (self.board == -me).T.astype(np.float64)
        if last_move[0] >= 0 and last_move[1] >= 0:
            state[2][last_move[1]][last_move[0]] = 1.0
        if current_player == 0:  # 轮到黑棋
            state[3][:,:] = 1.0
        return state
