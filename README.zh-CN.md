# gomoku_ai

[English](README.md) | **简体中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.10-ee4c2c.svg)](https://pytorch.org/)

一个五子棋（Gomoku / Five in a Row）的 AlphaZero 实现：**无锁大规模并行 C++ MCTS**
搜索引擎 + PyTorch 策略价值网络，完全从自我对弈中训练而来。内置 Web 界面，
可以直接在浏览器里与训练好的 AI 对弈。

开发日志（共 4 篇）：[(1) MCTS框架](https://www.douban.com/note/875904332/) ·
[(2) MCTS性能优化](https://www.douban.com/note/876037477/) ·
[(3) 11x11策略值网络模型的MCTS](https://www.douban.com/note/876852240/) ·
[(4) 15x15](https://www.douban.com/topic/496923907/)

## 演示

训练好的 `model_examples/15x15_snapshots/policy_game_9000` 在 15x15 棋盘上的一局
自对弈（每手 400 次 MCTS 模拟，16 核 CPU 上远低于 1 秒/手；GIF 由
`scripts/record_demo_gif.py` 生成）：

![15x15 训练模型自对弈](assets/selfplay_15x15.gif)

15x15 训练过程的曲线（1 万局自对弈，每 1000 局与另一个MCTS 对手评估一次）：

![训练曲线](assets/train_curve_15x15.png)

## 特性

- **C++ 无锁并行 MCTS**（`alphazero_mcts.hpp`）：棋盘状态用两个 `std::bitset`
  保存；子节点存放在排序的 `uint64_t` vector 中、打包进单个原子指针（高 16 位为
  ABA 计数）；每个节点的 并发数/访问次数/分数 打包进一个 8 字节原子变量；用
  **hazard pointer** 做内存回收；**virtual loss** 防止工作线程挤在相同节点上。
  通过 pybind11 暴露给 Python。
- **AlphaZero 训练管线**（`train.py`）：自对弈 -> 8 倍旋转/翻转数据增广 ->
  带 KL 自适应学习率的策略价值更新 -> 定期与固定对手评估，全程记录到
  tensorboard。
- **两代网络结构**：v1（3 层卷积，`policy_value_net_pytorch.py`）与 v2（ResNet，
  `policy_value_net_pytorch_v2.py`，默认）。加载时按权重内容自动识别 v1/v2。
- **浏览器对弈**：C++ Web 服务器（Crow + 内嵌 Python 即时转换模型）托管
  `templates/index.html`；AI 类型、模型、模拟次数、c_puct、线程数、执子颜色
  都可以在界面上按局调整。
- **评估工具**：`elo.py`（1v1 Elo 对战，带置信区间）与 `league.py`（多模型
  循环联赛，输出积分排名表与两两对战得分矩阵）。
- **棋盘尺寸**：8x8、9x9（纯 MCTS），11x11、15x15（纯 MCTS + AlphaZero）。

## 性能与棋力

以下数据均在 16 核 / 32 GB 的 x86-64 服务器上实测（详细过程与截图见上面的
开发日志）。

**纯 MCTS 引擎**

| 版本 | 棋盘 | 模拟次数 | 每手耗时 | 备注 |
|---|---|---|---|---|
| v0.1 | 8x8 | 1 千 | 0.01–0.05 秒 | 单线程；比向量化优化的 Python 版本快 20 倍以上 |
| v0.1 | 11x11 | 20 万 | 8–15 秒 | 大约小学低年级的业余水平 |
| v0.2 | 11x11 | 150 万 | 20–30 秒 | 20 线程 + jemalloc：搜索快约 7 倍，树内存回收快 10 倍以上；不错的业余水平，能识别各种双三 |

**AlphaZero（策略价值网络 + MCTS）**

| 模型 | 棋盘 | 网络 | 自对弈局数 | 训练耗时 | 棋力 |
|---|---|---|---|---|---|
| v0.3（`11x11_snapshots/policy_game_9000`） | 11x11 | ResNet 3 blocks / 64 ch | 1 万 | 约 20 小时 | 仅用 1~4 万次模拟即对 150 万次模拟的纯 MCTS 取得 20:0 全胜（1 万次约 2 秒/手）；1 千次模拟即与 50 万次纯 MCTS 打平 |
| `15x15_snapshots/policy_game_9000` | 15x15 | ResNet 3 blocks / 64 ch | 1 万 | 超过 24 小时 | 对 150 万次模拟的纯 MCTS 取得 20:0 全胜 |
| `15x15_6blocks_snapshots` | 15x15 | ResNet 6 blocks / 64 ch，2000 playouts | 1 万 | 约 65 小时 | 与 3 blocks 版本 1 万次模拟基本打平；**人类已经很难赢棋** |

值得注意的训练经验（来自开发日志）：

- 自对弈的瓶颈在模型推理：单线程 1 万次 rollout 的训练 5 小时只完成 14 局，
  必须多线程并行 rollout。
- 每个 worker 单 OMP 线程推理 + 导出时折叠 BatchNorm，自对弈速度提升约 10 倍。
- 把 Dirichlet 噪声从根节点选择挪到扩展阶段会直接导致训练坍缩；开局多样性
  至关重要。

## 目录结构

```
alphazero_mcts.hpp/.cpp    # 无锁并行 MCTS 核心 + pybind11 绑定（gomoku_ai 模块）
pure_mcts.hpp              # 早期的独立纯 MCTS 头文件，保留作参考
game.py                    # 对局循环（对战 / 自对弈数据收集）
player.py                  # 封装 C++ 引擎的 PureMCTSPlayer / AlphaZeroPlayer
policy_value_net_pytorch.py    # v1 策略价值网络（3 层卷积）
policy_value_net_pytorch_v2.py # v2 策略价值网络（ResNet，默认）；自动识别结构
train.py                   # 自对弈训练管线
elo.py                     # 两个玩家（模型或纯 MCTS）的 1v1 Elo 对战
league.py                  # N 个模型的循环联赛
plot_train_curve.py        # 从 tensorboard event 绘制 loss/entropy/评估曲线
web_server.cpp             # C++ Web 服务器（Crow）+ 内嵌 Python 模型转换
web.py                     # 早期的 Flask 演示服务器（已被 web_server 取代）
templates/index.html       # 浏览器对弈界面
model_examples/            # 训练好的模型快照（见下文）
scripts/record_demo_gif.py # 录制模型自对弈并生成 GIF（README 演示图的来源）
third_party/               # Crow（Web 框架）与 nlohmann/json
setup.py, install_plugin.sh      # 编译 gomoku_ai Python 扩展
compile_web_server.sh            # 编译 Web 服务器
```

## 环境依赖

- Linux、**clang++**（必需：g++ -O3 的严格别名优化会误编译 MCTS 树中的指针
  打包代码，详见 `setup.py` 注释）、Python >= 3.9
- Python 包：`pip3 install -r requirements.txt`
- C++ 库：libtorch（随 pip 版 `torch` 包自带）、**folly**、gflags、glog、fmt、
  libunwind、double-conversion、libiberty、libevent、boost-context、jemalloc、
  libgomp

Debian/Ubuntu 下可以这样安装 C++ 依赖：

```bash
apt-get install libfolly-dev libgflags-dev libgoogle-glog-dev libfmt-dev \
    libunwind-dev libdouble-conversion-dev libiberty-dev libevent-dev \
    libboost-context-dev libjemalloc-dev libgomp1
```

## 编译

```bash
pip3 install -r requirements.txt

./install_plugin.sh        # 编译 gomoku_ai.*.so（C++ MCTS Python 扩展）
./compile_web_server.sh    # 编译 ./web_server
```

两个脚本都会从当前解释器自动探测 torch / pybind11 / python 路径，因此
virtualenv 或其他 Python 版本也能直接使用。

## 使用方法

### 在浏览器中与 AI 对弈

```bash
# 11x11，使用训练好的快照（state_dict 会自动转换为 TorchScript）：
./web_server -m model_examples/11x11_snapshots/policy_game_9000.model -n 10000

# 15x15，使用 6-block ResNet 快照：
./web_server -s 15 -m model_examples/15x15_6blocks_snapshots/policy_game_9000.model -n 10000

# 或者不加载模型，使用纯 MCTS：
./web_server -n 1000000
```

然后打开 <http://localhost:7000>。AI 类型、模型、模拟次数、c_puct、线程数与
执子颜色都可以在 Web 界面上按局修改。运行 `./web_server --help` 查看全部选项
（搜索线程数默认取本机 CPU 核数）。

### 从零开始训练

```bash
./train.py                                   # 11x11，v2 ResNet 3 blocks / 64 通道
./train.py --board-size 15 --num-blocks 6 --channels 64 --n-playout 2000
./train.py current_policy.ckpt               # 从 checkpoint 续训
```

训练产物（checkpoint `current_policy*.ckpt`、导出的模型
`current_policy*.model`、tensorboard 目录 `gomoku_experiments*/`，以及每次评估
保存到 `<board>x<board>_snapshots/` 的快照）都按棋盘尺寸带后缀，11x11 与 15x15
的训练任务不会互相覆盖。可以定期与纯 MCTS 或固定模型（`--opp-model`）评估；
用以下命令查看曲线：

```bash
./plot_train_curve.py --dir ./gomoku_experiments
tensorboard --logdir ./gomoku_experiments
```

### 评估模型

```bash
# AlphaZero 模型 vs 纯 MCTS（默认对战组合）：
./elo.py -n 20 --a-model model_examples/11x11_snapshots/policy_game_9000.model \
         --b-model= --b-simulations 500000

# 模型 vs 模型：
./elo.py -n 20 --a-model a.model --b-model b.model

# 对示例快照跑循环联赛：
./league.py --games 10 --models=$(ls model_examples/11x11_snapshots/*.model | tr '\n' ',')
```

### 在 Python 中调用引擎

```python
import torch        # 必须先于 gomoku_ai 导入（见 elo.py 中的说明）
import gomoku_ai

# 11x11 纯 MCTS，工作线程数建议取 CPU 核数：
game = gomoku_ai.PureMCTSFramework11(cores=16, c_puct=2.0, reuse_tree_states=True)
move = game.SearchBestMove(simulate_times=100000)   # -> (x, y)

# AlphaZero MCTS（TorchScript 模型）：
game = gomoku_ai.AlphaZeroMCTSFramework15(cores=16, c_puct=5.0, reuse_tree_states=True)
moves, probs = game.SearchBestMove(simulate_times=1000, model_path='model.pt', temperature=1.0)
```

绑定类名为 `PureMCTSFramework{8,9,11,15}` 与 `AlphaZeroMCTSFramework{8,11,15}`。

## 模型示例

`model_examples/` 下提供了预训练快照（每 1000 局自对弈保存一个）：

| 目录 | 棋盘 | 网络 | 备注 |
|---|---|---|---|
| `11x11_snapshots/` | 11x11 | ResNet 3 blocks / 64 ch | `policy_game_9000` 是该轮训练中最强的 |
| `15x15_snapshots/` | 15x15 | ResNet 3 blocks / 64 ch | `policy_game_9000` 对 150 万次模拟纯 MCTS 20:0 全胜 |
| `15x15_6blocks_snapshots/` | 15x15 | ResNet 6 blocks / 64 ch | 每手 2000 次 playout 训练；最强的系列 |

它们都是 PyTorch `state_dict`（`.model`）；`elo.py`、`league.py` 与
`web_server` 会在加载时自动转换成 TorchScript。

## 致谢

- [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) ——
  本项目最初参考的实现
- [Crow](https://github.com/CrowCpp/Crow) 与
  [nlohmann/json](https://github.com/nlohmann/json)（vendor 在 `third_party/` 下）、
  [Folly](https://github.com/facebook/folly)、[LibTorch](https://pytorch.org/)、
  [pybind11](https://github.com/pybind/pybind11)
- 论文：*Mastering the game of Go without human knowledge*（AlphaGo Zero）与
  *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning
  Algorithm*（AlphaZero）

## 许可证

[MIT](LICENSE)
