#! /usr/bin/python3

from datetime import datetime
import gomoku_ai
from flask import Flask, request, jsonify, render_template, session
import torch
from flask_cors import CORS
import numpy as np
import torch.nn as nn

app = Flask(__name__)
#app.secret_key = 'your_secret_key'  # 设置一个密钥用于session
CORS(app)

# 初始化棋盘
@app.route('/')
def index():
    return render_template('./index.html')

CORES = 1
#game = gomoku_ai.PureMCTSGame11(CORES, 5.0, False)
game = None

@app.route('/handle_state', methods=['POST'])
def move():
    global game
    data = request.json
    boardArr = data['board']
    x = data['x']
    y = data['y']

    # 1是黑棋，-1是白棋，0是空
    print(f'Human move: {x}, {y}')
    boardArr[x][y] = 0
    np_board = np.array(boardArr, dtype=np.int32)
    if (not game) or (not game.StateEquals(np_board, False)):
        if np.array_equal(np_board, np.zeros((11, 11), dtype=np.int32)):
            print("Initializing a new game!")
        else:
            print("WARING: Re-Initializing the game unexpectedly!")
        np_board[x][y] = 1
        game = gomoku_ai.PureMCTSGame11(CORES, np_board, (x, y), 2.0, True)
    else:
        game.Play(x, y)

    if game.AvailableCount() == 0:
        print("Draw!")
        return jsonify({'result' : 'draw'})
    if game.IsEnd():
        print("Human win!")
        return jsonify({'result' : 'win'})

    #search_times = 200000
    search_times = 200000
    start_time = datetime.now()
    ai_x, ai_y = game.SearchBestMove(search_times)
    time_cost = (datetime.now() - start_time).total_seconds()
    print(f"MCTS time cost: {time_cost:.2f} seconds, search count:{search_times}")
    print(f"AI move:({ai_x},{ai_y})")
    game.Play(ai_x, ai_y)
    if game.AvailableCount() == 0:
        print("Draw!")
        return jsonify({'result' : 'draw', 'ai_move' : [ai_x, ai_y]})
    if game.IsEnd():
        print("AI win!")
        return jsonify({'result' : 'lose', 'ai_move' : [ai_x, ai_y]})

    return jsonify({'result': 'continue', 'ai_move': [ai_x, ai_y]})

@app.route('/restart', methods=['POST'])
def restart():
    return jsonify({'result': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=7000)

