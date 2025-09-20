#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct High8Compare {
    bool operator()(uint64_t a, uint64_t b) const noexcept {
        return (a & 0xFF00000000000000) < (b & 0xFF00000000000000);
    }
};

// find the first >= target iterator.
auto find_insertion_point(std::vector<uint64_t>& vec, uint8_t key) {
    uint64_t key_pattern = static_cast<uint64_t>(key) << 56;
    return std::lower_bound(vec.begin(), vec.end(), key_pattern, High8Compare());
}

bool try_insert_sorted(std::vector<uint64_t>& vec, uint64_t value) {
    uint8_t key = value >> 56;
    auto it = find_insertion_point(vec, key);
    
    if ((it != vec.end() && (*it >> 56) == key)) {
        return false;
    }
    
    vec.insert(it, value);
    return true;
}

auto find_by_key(const std::vector<uint64_t>& vec, uint8_t key) {
    uint64_t key_pattern = static_cast<uint64_t>(key) << 56;
    auto it = std::lower_bound(vec.begin(), vec.end(), key_pattern, High8Compare());
    
    if (it != vec.end() && (*it >> 56) == key) {
        return it;
    }
    return vec.end();
}

namespace gomoku_ai {
typedef std::pair<int, int> Move;  // first->x,second->y
std::string format(const char* fmt, ...) {
    va_list args;
    
    va_start(args, fmt);
    int length = vsnprintf(nullptr, 0, fmt, args); // C++11 标准规定，若buf为nullptr且size为0，则返回所需字节数（不含空终止符）
    va_end(args);
    
    if (length <= 0) {
        return ""; // 格式化错误，返回空字符串
    }
    
    size_t buf_size = length + 1;
    std::vector<char> buf(buf_size);
    
    va_start(args, fmt);
    vsnprintf(buf.data(), buf_size, fmt, args); // 使用vector的data()成员函数获取裸指针
    va_end(args);
    
    return std::string(buf.data());
}

template <int BOARD_SIZE>
struct TreeNode {
    TreeNode* parent_;
    // std::map<int, TreeNode*> children_;
    std::vector<uint64_t> children_;
    std::bitset<BOARD_SIZE * BOARD_SIZE> availables_;
    std::bitset<BOARD_SIZE * BOARD_SIZE> blacks_;
    int visits_ = 0;
    int scores_ = 0;
    // float p = 1.0;  // prior probability

    TreeNode() : parent_(nullptr) {
        availables_.flip();
    }

    TreeNode(TreeNode* parent, int move_index, bool is_last_black) : parent_(parent), availables_(parent->availables_), blacks_(parent->blacks_) {
        availables_.set(move_index, false);
        blacks_.set(move_index, is_last_black);
    }

    ~TreeNode() {
        // 不管parent的内存
        for (auto itr = children_.begin(); itr != children_.end(); ++itr) {
            auto ptr = (TreeNode*)((*itr) & 0x00ffffffffffffff);
            delete ptr;
        }
    }
    
    bool IsEnd(Move last_move, bool is_last_black) const {
        if (availables_.count() == 0) {
            return true;
        }
        if (last_move.first < 0 || last_move.second < 0) {
            return false;
        }

        auto not_available = availables_;
        not_available.flip();
        std::bitset<BOARD_SIZE*BOARD_SIZE> solid = blacks_;
        if (!is_last_black) {
            solid.flip();
        }
        solid &= not_available;
        int idx = last_move.second * BOARD_SIZE + last_move.first;

        // 横向
        int diff_begin = 0;
        while (last_move.first - diff_begin >= 0 && solid[idx - diff_begin]) { ++diff_begin; }
        int diff_end = 0;
        while (last_move.first + diff_end < BOARD_SIZE && solid[idx + diff_end]) { ++diff_end; }
        if (diff_end + diff_begin - 1 >= 5) {
            return true;
        }

        // 竖向
        diff_begin = 0;
        while (last_move.second - diff_begin >= 0 && solid[idx - diff_begin * BOARD_SIZE]) { ++diff_begin; }
        diff_end = 0;
        while (last_move.second + diff_end < BOARD_SIZE && solid[idx + diff_end * BOARD_SIZE]) { ++diff_end; }
        if (diff_end + diff_begin - 1 >= 5) {
            return true;
        }

        // 斜向1
        diff_begin = 0;
        while (last_move.first - diff_begin >= 0 && last_move.second - diff_begin >= 0 && solid[idx - diff_begin * (BOARD_SIZE + 1)]) { ++diff_begin; }
        diff_end = 0;
        while (last_move.first + diff_end < BOARD_SIZE && last_move.second + diff_end < BOARD_SIZE && solid[idx + diff_end * (BOARD_SIZE + 1)]) { ++diff_end; }
        if (diff_end + diff_begin - 1 >= 5) {
            return true;
        }

        // 斜向2
        diff_begin = 0;
        while (last_move.first - diff_begin >= 0 && last_move.second + diff_begin < BOARD_SIZE && solid[idx + diff_begin * (BOARD_SIZE - 1)]) { ++diff_begin; }
        diff_end = 0;
        while (last_move.first + diff_end < BOARD_SIZE && last_move.second - diff_end >= 0 && solid[idx - diff_end * (BOARD_SIZE - 1)]) { ++diff_end; }
        if (diff_end + diff_begin - 1 >= 5) {
            return true;
        }

        return false;
    }        

    auto SelectAndExpand(std::mt19937& engine, float c_puct, bool is_black) {
        float max_q = -1.0;
        std::vector<int> candidates;
        for (int i = 0; i < BOARD_SIZE* BOARD_SIZE; ++i) {
            if (!availables_[i]) {
                continue;
            }

            auto itr = find_by_key(children_, (uint8_t)i);
            float q = 0;
            float puct = c_puct * sqrt(visits_);
            if (itr != children_.end()) {
                auto* child = (TreeNode*)(*itr & 0x00ffffffffffffff);
                q = (float)child->scores_ / child->visits_;
                puct /= (child->visits_ + 1);
            }
            auto adjust_q = q + puct;
            if (adjust_q > max_q) {
                candidates.clear();
                max_q = adjust_q;
            }
            if (adjust_q >= max_q-1e-5) {
                candidates.emplace_back(i);
            }
        }

        std::uniform_int_distribution<int> distribution(0, (int)candidates.size() - 1);
        int child_idx = candidates[distribution(engine)];
        auto itr = find_insertion_point(children_, (uint8_t)child_idx);
        if (itr == children_.end() || (*itr >> 56) != (uint64_t)child_idx) {
            auto* entry = new TreeNode<BOARD_SIZE>(this, child_idx, is_black);
            auto elem = ((uint64_t)entry | ((uint64_t)child_idx << 56));
            return children_.insert(itr, elem);
        } else {
            return itr;
        }
    }

    void UpdateRecursively(int score) {
        if (parent_ != nullptr) {
            parent_->UpdateRecursively(score);
        }
        ++visits_;
        scores_ += score;
    }
};

template <int BOARD_SIZE = 15>
class PureMCTSGame {
public:
    PureMCTSGame(float c_puct, bool reuse_tree_states = false) : root_(new TreeNode<BOARD_SIZE>()), last_move_{-1, -1}, c_puct_(c_puct), is_last_black_(false), reuse_tree_states_(reuse_tree_states) {}

    PureMCTSGame(py::array_t<int32_t> board, Move last_move, float c_puct, bool reuse_tree_states = false) : root_(new TreeNode<BOARD_SIZE>()), last_move_(last_move), c_puct_(c_puct), reuse_tree_states_(reuse_tree_states) {
        py::buffer_info buffer_info = board.request();
        if (buffer_info.ndim != 2) {
            throw std::runtime_error("Number of dimension must be 2");
        }
        if (buffer_info.shape[0] != BOARD_SIZE || buffer_info.shape[1] != BOARD_SIZE) {
            throw std::runtime_error(format("Board must be %dx%d!", BOARD_SIZE, BOARD_SIZE));
        }
        if (last_move_.first >= BOARD_SIZE || last_move_.second >= BOARD_SIZE) {
            throw std::runtime_error(format("Last move(%d,%d) is out of board(board size:%d)!", last_move.first, last_move.second, BOARD_SIZE));
        }

        int32_t* ptr = static_cast<int32_t*>(buffer_info.ptr);
        for (int y = 0; y < BOARD_SIZE; ++y) {
           for (int x = 0; x < BOARD_SIZE; ++x) {
               int offset = x * buffer_info.strides[0] / sizeof(int32_t) + y * buffer_info.strides[1] / sizeof(int32_t);
               int index = y * BOARD_SIZE + x;
               if (ptr[offset] == 1) {
                   root_->availables_.set(index, false);
                   root_->blacks_.set(index, true);
               } else if (ptr[offset] == -1) {
                   root_->availables_.set(index, false);
                   root_->blacks_.set(index, false);
               } else if (ptr[offset] == 0) {
                   root_->availables_.set(index, true);
                   root_->blacks_.set(index, false);
               } else {
                   throw std::runtime_error(format("Board(%d,%d) = %d is not in [-1, 0, 1]!", x, y, ptr[offset]));
               }
           }
        }
        if (last_move_.first < 0 || last_move_.second < 0) {
            if (root_->availables_.count() != BOARD_SIZE * BOARD_SIZE) {
                throw std::runtime_error(format("NO last action but there are %d picies on board!", BOARD_SIZE * BOARD_SIZE - root_->availables_.count()));
            }
            is_last_black_ = false;  // 强要求黑棋先走
        } else {
            int index = last_move_.second * BOARD_SIZE + last_move_.first;
            if (root_->availables_[index]) {
                throw std::runtime_error(format("The move(%d,%d) hasnot been made!", last_move.first, last_move.second));
            }
            is_last_black_ = root_->blacks_[index];
        }
    }

    bool StateEquals(py::array_t<int32_t> board, bool is_last_black) const {
        if (is_last_black != is_last_black_) {
            return false;
        }

        py::buffer_info buffer_info = board.request();
        int32_t* ptr = static_cast<int32_t*>(buffer_info.ptr);
        auto availables = root_->availables_;
        auto blacks = root_->blacks_;
        for (int y = 0; y < BOARD_SIZE; ++y) {
           for (int x = 0; x < BOARD_SIZE; ++x) {
               int offset = x * buffer_info.strides[0] / sizeof(int32_t) + y * buffer_info.strides[1] / sizeof(int32_t);
               int index = y * BOARD_SIZE + x;
               if (ptr[offset] == 1) {
                   availables.set(index, false);
                   blacks.set(index, true);
               } else if (ptr[offset] == -1) {
                   availables.set(index, false);
                   blacks.set(index, false);
               } else if (ptr[offset] == 0) {
                   availables.set(index, true);
                   blacks.set(index, false);
               } else {
                   throw std::runtime_error(format("Board(%d,%d) = %d is not in [-1, 0, 1]!", x, y, ptr[offset]));
               }
           }
        }
        return (availables == root_->availables_ && blacks == root_->blacks_); 
    }

    int AvailableCount() const {
        return (int)root_->availables_.count();
    }

    void Play(int x, int y) {
        TreeNode<BOARD_SIZE>* new_root = nullptr;
        int child_idx = y * BOARD_SIZE + x;
        if (!root_->availables_[child_idx]) {
            throw std::runtime_error(format("(%d,%d) is not avialable!", x, y));
        }
        if (!reuse_tree_states_) {
            new_root = new TreeNode<BOARD_SIZE>(root_, child_idx, !is_last_black_);
        } else {
            auto itr = find_insertion_point(root_->children_, (uint8_t)child_idx);
            if (itr == root_->children_.end() || (*itr >> 56) > (uint64_t)child_idx) {
                 auto* entry = new TreeNode<BOARD_SIZE>(root_, child_idx, !is_last_black_);
                 auto elem = ((uint64_t)entry | ((uint64_t)child_idx << 56));
                 root_->children_.insert(itr, elem);
            }
            auto it = find_by_key(root_->children_, (uint8_t)child_idx);
            new_root = (TreeNode<BOARD_SIZE>*)(*it & 0x00ffffffffffffff);
            root_->children_.erase(it);
        }
        new_root->parent_ = nullptr;
        delete root_;
        root_ = new_root;
        last_move_ = { x, y };
        is_last_black_ = !is_last_black_;
   }

    bool IsEnd() const {
        return root_->IsEnd(last_move_, is_last_black_);
    }

    Move SearchBestMove(int simulate_times);
private:
    TreeNode<BOARD_SIZE>* root_;
    Move last_move_;
    float c_puct_;
    bool is_last_black_{false};
    bool reuse_tree_states_;
};

template <int BOARD_SIZE>
Move PureMCTSGame<BOARD_SIZE>::SearchBestMove(int simulate_times) {
    static std::random_device rd;
    static std::mt19937 engine(rd());

    Move ret = { -1, -1 };
    if (root_->IsEnd(last_move_, is_last_black_)) {
        return ret;
    }

    for (int i = 0; i < simulate_times; ++i) {
        if ((i % 100000) == 0) {
            std::cout << "Simulate process: " << i << " / " << simulate_times << std::endl;
        }
        TreeNode<BOARD_SIZE>* node = root_;
        Move last_move = last_move_;
        bool is_last_black = is_last_black_;
        while (!node->IsEnd(last_move, is_last_black)) {  // step 3: simulate
            auto itr = node->SelectAndExpand(engine, c_puct_, !is_last_black);  // step 1&2: expand->select
            auto idx = (int)(*itr >> 56);
            last_move = { (idx % BOARD_SIZE), (idx / BOARD_SIZE) };
            is_last_black = !is_last_black;
            node = (TreeNode<BOARD_SIZE>*)(*itr & 0x00ffffffffffffff);
        }
        int score = 0;
        if (node->availables_.count() > 0) {
            score = (is_last_black != is_last_black_) ? 1 : -1;
        }
        node->UpdateRecursively(score);  // step 4: backpropagate
    }

    // find the most visited.
    std::vector<Move> best_actions;
    int max_visits = 0;
    for (auto itr = root_->children_.begin(); itr != root_->children_.end(); ++itr) {
        auto* child = (TreeNode<BOARD_SIZE>*)(*itr & 0x00ffffffffffffff);
        auto idx = (int)(*itr >> 56);
        if (child->visits_ > max_visits) {
            max_visits = child->visits_;
            best_actions.clear();
        }
        if (child->visits_ == max_visits) {
            best_actions.emplace_back(Move((idx % BOARD_SIZE), idx / BOARD_SIZE));
        }
    }

    std::uniform_int_distribution<int> distribution(0, (int)best_actions.size() - 1);
    return best_actions[distribution(engine)];
}
}  // namespace gomoku_ai

PYBIND11_MODULE(gomoku_ai, m) {
    m.doc() = "Pybind11 pure MCTS gomoku plugin";

    py::class_<gomoku_ai::PureMCTSGame<8>>(m, "PureMCTSGame8")
        .def(py::init<float, bool>(),
            py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def(py::init<py::array_t<int32_t>, gomoku_ai::Move, float, bool>(),
            py::arg("board"), py::arg("last_move"), py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    board(numpy(int32)): the board state (with last move)\n"
            "    last_move((int,int)): last move coordination\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def("StateEquals", &gomoku_ai::PureMCTSGame<8>::StateEquals, py::arg("board"), py::arg("s_last_black"), "Is the state equals the game's state")
        .def("Play", &gomoku_ai::PureMCTSGame<8>::Play, py::arg("x"), py::arg("y"),
            "Make a move.\n")
        .def("AvailableCount", &gomoku_ai::PureMCTSGame<8>::AvailableCount, "How many grids there are to put pieces\n")
        .def("IsEnd", &gomoku_ai::PureMCTSGame<8>::IsEnd, "Is the game over(win,lose or draw)\n")
        .def("SearchBestMove", &gomoku_ai::PureMCTSGame<8>::SearchBestMove, py::arg("simulate_times"), "Monto-Carlo tree search for specified times to search the best move.\n");

    py::class_<gomoku_ai::PureMCTSGame<9>>(m, "PureMCTSGame9")
        .def(py::init<float, bool>(),
            py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def(py::init<py::array_t<int32_t>, gomoku_ai::Move, float, bool>(),
            py::arg("board"), py::arg("last_move"), py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    board(numpy(int32)): the board state (with last move)\n"
            "    last_move((int,int)): last move coordination\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def("StateEquals", &gomoku_ai::PureMCTSGame<9>::StateEquals, py::arg("board"), py::arg("s_last_black"), "Is the state equals the game's state")
        .def("Play", &gomoku_ai::PureMCTSGame<9>::Play, py::arg("x"), py::arg("y"),
            "Make a move.\n")
        .def("AvailableCount", &gomoku_ai::PureMCTSGame<9>::AvailableCount, "How many grids there are to put pieces\n")
        .def("IsEnd", &gomoku_ai::PureMCTSGame<9>::IsEnd, "Is the game over(win,lose or draw)\n")
        .def("SearchBestMove", &gomoku_ai::PureMCTSGame<9>::SearchBestMove, py::arg("simulate_times"), "Monto-Carlo tree search for specified times to search the best move.\n");

    py::class_<gomoku_ai::PureMCTSGame<11>>(m, "PureMCTSGame11")
        .def(py::init<float, bool>(),
            py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def(py::init<py::array_t<int32_t>, gomoku_ai::Move, float, bool>(),
            py::arg("board"), py::arg("last_move"), py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    board(numpy(int32)): the board state (with last move)\n"
            "    last_move((int,int)): last move coordination\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def("StateEquals", &gomoku_ai::PureMCTSGame<11>::StateEquals, py::arg("board"), py::arg("s_last_black"), "Is the state equals the game's state")
        .def("Play", &gomoku_ai::PureMCTSGame<11>::Play, py::arg("x"), py::arg("y"),
            "Make a move.\n")
        .def("AvailableCount", &gomoku_ai::PureMCTSGame<11>::AvailableCount, "How many grids there are to put pieces\n")
        .def("IsEnd", &gomoku_ai::PureMCTSGame<11>::IsEnd, "Is the game over(win,lose or draw)\n")
        .def("SearchBestMove", &gomoku_ai::PureMCTSGame<11>::SearchBestMove, py::arg("simulate_times"), "Monto-Carlo tree search for specified times to search the best move.\n");

    py::class_<gomoku_ai::PureMCTSGame<15>>(m, "PureMCTSGame15")
        .def(py::init<float, bool>(),
            py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def(py::init<py::array_t<int32_t>, gomoku_ai::Move, float, bool>(),
            py::arg("board"), py::arg("last_move"), py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    board(numpy(int32)): the board state (with last move)\n"
            "    last_move((int,int)): last move coordination\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def("StateEquals", &gomoku_ai::PureMCTSGame<15>::StateEquals, py::arg("board"), py::arg("s_last_black"), "Is the state equals the game's state")
        .def("Play", &gomoku_ai::PureMCTSGame<15>::Play, py::arg("x"), py::arg("y"),
            "Make a move.\n")
        .def("AvailableCount", &gomoku_ai::PureMCTSGame<15>::AvailableCount, "How many grids there are to put pieces\n")
        .def("IsEnd", &gomoku_ai::PureMCTSGame<15>::IsEnd, "Is the game over(win,lose or draw)\n")
        .def("SearchBestMove", &gomoku_ai::PureMCTSGame<15>::SearchBestMove, py::arg("simulate_times"), "Monto-Carlo tree search for specified times to search the best move.\n");
}

