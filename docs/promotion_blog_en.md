# Building a Gomoku AlphaZero from Scratch: a Lock-Free Parallel MCTS in C++, and Everything That Broke

*Draft for dev.to / Medium / personal blog. Adapt the title and links as needed.*

---

**TL;DR** — I wrote an AlphaZero for Gomoku (Five in a Row) from scratch: a
lock-free, multithreaded Monte Carlo Tree Search engine in C++ (with hazard
pointers, bit-packed atomics and virtual loss), a PyTorch policy-value
network, and a self-play training pipeline. On a 16-core CPU server, the
final 15x15 model beats a pure MCTS player doing **1.5 million** rollouts per
move while using only **1,000–40,000** simulations itself — about 2 seconds
per move. Code: https://github.com/bryanzhang/gomoku_ai

![Self-play of the trained 15x15 model](../assets/selfplay_15x15.gif)

## Why

Reinforcement learning is notoriously fragile compared to supervised
fine-tuning, and AlphaZero is one of those algorithms everyone understands in
principle and almost nobody has actually run end-to-end. I wanted to build
one from scratch — no RL framework, no game engine library — and Gomoku is
the perfect testbed: simple rules, small action space, but real enough that
you can feel the AI getting stronger game by game.

The plan had two stages, and each became an engineering project of its own:

1. Build a *fast* pure (model-free) MCTS that already plays at a decent
   amateur level. MCTS speed directly bounds training speed later, because
   the agent *is* the search.
2. Bolt a policy-value network onto it and run the AlphaZero self-play loop.

## Stage 1: How fast can pure MCTS get?

The search core is C++ exposed to Python via pybind11. The first version was
already ~20x faster than a vectorized Python implementation (1k rollouts on
8x8: 0.01–0.05 s), but a single thread left the other 15 cores of the
machine idle. Making tree search parallel is where the fun started.

The final design:

- **Board state**: two `std::bitset`s (empty/occupied and color), so copying
  a node is copying two words.
- **Children storage**: a sorted `std::vector<uint64_t>`, one word per
  child — pointer in the low 48 bits, move index in the top 8 bits.
- **One atomic per node for the whole search state**: the top byte counts
  how many threads are currently inside the node (concurrency), the next
  three bytes hold the visit count, and the low four bytes hold the
  accumulated score. Selection, expansion and backpropagation are all CAS
  loops on this single word.
- **ABA counter**: the children pointer is atomically swapped on expansion;
  the top 16 bits of the packed word carry a version counter so a CAS that
  "sees the same pointer" can still detect a replace.
- **Hazard pointers** protect readers of the children vector while another
  thread replaces it; retired vectors are reclaimed in batches, per thread.
- **Virtual loss** discourages threads from piling onto the same node.

On a 16-core box with 20 worker threads and jemalloc this was ~7x faster
than the single-threaded version, and parallelizing *tree reclamation* (the
`Play` path had become the bottleneck once search was fast) cut move latency
by another 10x. Net result: 1.5M rollouts on 11x11 in 20–30 seconds, playing
at a solid amateur level — it reliably spots double-three forks.

Two bugs from this phase deserve special mention because of how they failed:

- In the hazard-pointer acquire loop, the `expected` value of the CAS must
  be re-initialized **inside** the loop. A failed CAS overwrites it with the
  current value (`true`); on the next iteration you compare `true` against
  another thread's active record and the CAS "succeeds". Two threads share
  one hazard record, protection silently vanishes, a retired children vector
  gets freed early, and the next reader dereferences a wild pointer —
  eventually exploding the stack inside `uniform_int_distribution(0, -1)`.
  Found via a core dump after 8,000 training games.
- The hand-written pointer packing is miscompiled by `g++ -O3` strict
  aliasing. The project must be built with `clang++`; `setup.py` forces it.

## Stage 2: AlphaZero on top

With the engine in place, the algorithm is textbook: self-play games produce
(state, visit-count distribution, outcome) triples; the network is trained
to match them; the improved network immediately drives the next round of
self-play. The reality was a year-long series of wall-clock problems.

**Inference dominates everything.** A naive single-threaded run did 14
self-play games in 5 hours — each of the 10k rollouts per move is a network
forward pass. Multithreading the rollouts (20 threads, 20k rollouts) made it
5x faster but visibly destabilized convergence.

**Exploration details matter more than you think.** At one point I moved the
Dirichlet noise from root selection into node expansion; training collapsed
almost immediately. Rolling back recovered it. Likewise, the value target is
only clean if the endgame is played greedily: after the first ~15 plies of
temperature-1 exploration, self-play switches to near-argmax moves.

**Two infrastructure fixes gave ~10x.** First, every worker thread must pin
its intra-op parallelism to 1 (`at::set_num_threads(1)` plus
`omp_set_num_threads(1)` — the latter is needed because the THNN
`slow_conv2d` path doesn't read ATen's setting). 16 workers each spawning 16
OMP threads on 16 cores made a 1000-simulation move 5x *slower*. Second,
BatchNorm is folded into the preceding convolutions at export time, so the
exported TorchScript graph is BN-free — equivalent to `torch.jit.freeze`,
but done manually because repeated `trace`/`freeze` calls leak megabytes of
C++ memory per call, and the pipeline exports the model once per game.

One more hard-won detail: the exported model file must be written
**atomically** (`tmp` + `rename`). The C++ side watches the file's mtime and
reloads it mid-game; a worker reading a half-written file throws a
`c10::Error` inside a folly worker, which cannot unwind safely and aborts
the whole process.

## Results

All numbers from a 16-core / 32 GB x86-64 server, CPU only.

| Model | Board | Network | Self-play games | Wall time | Strength |
|---|---|---|---|---|---|
| pure MCTS v0.2 | 11x11 | — | — | — | 1.5M rollouts/move in 20–30 s; strong amateur |
| 11x11 `policy_game_9000` | 11x11 | ResNet 3 blocks / 64 ch | 10k | ~20 h | **20:0** vs 1.5M-rollout pure MCTS with 10k–40k sims; ties 500k-rollout pure MCTS at 1k sims |
| 15x15 `policy_game_9000` | 15x15 | ResNet 3 blocks / 64 ch | 10k | >24 h | **20:0** vs 1.5M-rollout pure MCTS |
| 15x15 6-block series | 15x15 | ResNet 6 blocks / 64 ch, 2000 playouts | 10k | ~65 h | clearly above the 3-block series; very hard for a human to beat |

![Training curves](../assets/train_curve_15x15.png)

The moment that made the year of debugging worth it: the best 11x11
checkpoint, given *one thousand* simulations per move (~2 seconds), played
twenty games against the pure-MCTS engine doing *1.5 million* rollouts per
move — and won all twenty.

## Try it

```bash
pip3 install -r requirements.txt
./install_plugin.sh && ./compile_web_server.sh
./web_server -s 15 -m model_examples/15x15_6blocks_snapshots/policy_game_10000.model
# open http://localhost:7000
```

Pretrained snapshots (one per 1000 self-play games) ship in
[`model_examples/`](https://github.com/bryanzhang/gomoku_ai/tree/master/model_examples);
the repo also includes an Elo match tool (`elo.py`), a round-robin league
runner (`league.py`) and the full training pipeline (`train.py`) if you want
to train your own.

If this was useful or interesting, a star on
[GitHub](https://github.com/bryanzhang/gomoku_ai) is much appreciated — and
issues/PRs are welcome. The detailed Chinese dev log is on Douban:
[(1)](https://www.douban.com/note/875904332/)
[(2)](https://www.douban.com/note/876037477/)
[(3)](https://www.douban.com/note/876852240/)
[(4)](https://www.douban.com/topic/496923907/).
