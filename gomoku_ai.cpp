#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <vector>
#include <random>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/Unit.h>
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

void invalidate_cache_region_atomic(void* data_ptr, size_t size) {
    auto* masked_data_ptr = (void*)((uint64_t)data_ptr & 0xffffffffffffff40uLL);
    auto adjusted_size = size + ((uint8_t*)data_ptr - (uint8_t*)masked_data_ptr);
    auto* ptr = static_cast<std::atomic<uintptr_t>*>(masked_data_ptr);
    size_t count = (adjusted_size + 63) / 64;
    
    for (size_t i = 0; i < count; ++i) {
        // 原子操作可能生成lock指令
        ptr[i * (64 / sizeof(uintptr_t))].fetch_or(0, std::memory_order_acq_rel);
    }
    
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void MakeChildrenVisible(std::vector<uint64_t>& children) {
    invalidate_cache_region_atomic((void*)children.data(), children.size() * sizeof(uint64_t));
    invalidate_cache_region_atomic(&children, sizeof(children));
}

void MakeAllVisible() {
    std::mutex mut;
    std::lock_guard lock(mut);
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

// Hazard pointer
template <class TYPE>
struct HPRecType {
   std::atomic<HPRecType*> next_ = nullptr;
   std::atomic<bool> active_ = false;
   std::atomic<TYPE*> hazard_ = nullptr;  // NOTE(junhaozhang): 并不own!

   void Release() {
        hazard_ = nullptr;
        active_ = false;
   }
};

// Hazard pointer list
template <class TYPE, size_t R = 1000>
class HPList {
public:
    HPRecType<TYPE>* Acquire() {
       HPRecType<TYPE>* p = head_;
       bool expected_false = false;
       for (; p; p = p->next_) {
           if (!p->active_.compare_exchange_strong(expected_false, true)) {
               continue;
           }
           return p;
       }

       // increment the list length
       p = new HPRecType<TYPE>();
       p->active_ = true;
       HPRecType<TYPE>* old;
       do {
           old = head_;
           p->next_ = old;
       } while (!head_.compare_exchange_strong(old, p));
       return p;
    }

    ~HPList() {
        Clear();
    }

    void Clear() {
        HPRecType<TYPE>* p = head_;
        while (p) {
            HPRecType<TYPE>* next = p->next_;
            delete p;
            p = next;
        }
        head_ = nullptr;
    }

    void Retire(std::vector<TYPE*>& retire_list, TYPE* p) {
        retire_list.push_back(p);
        if (retire_list.size() < R) {
            return;
        }
        TryReclaim(retire_list);
    }

    void TryReclaim(std::vector<TYPE*>& retire_list) {
        // stage 1: scan hazard pointers list
        // collecting all non-null ptrs
        std::set<TYPE*> hp;
        HPRecType<TYPE>* head = head_;
        while (head) {
            TYPE* p = head->hazard_;
            if (p) {
                hp.insert(p);
            }
            head = head->next_;
        }
        
        // stage 2: reclaim pointers that are not in hp list.
        typename std::vector<TYPE*>::iterator itr = retire_list.begin();
        while (itr != retire_list.end()) {
            if (hp.find(*itr) == hp.end()) {
                delete *itr;
                *itr = retire_list.back();
                retire_list.pop_back();
            } else {
                ++itr;
            }
        }
    }

    void print_list() {
        HPRecType<TYPE>* p = head_;
        std::cerr << "HP_LIST: ";
        while (p) {
            std::cerr << p->active_ << "|" << p->hazard_ << " -> ";
            p = p->next_;
        }
        std::cerr << std::endl;
    }

private:
    std::atomic<HPRecType<TYPE>*> head_ = nullptr;
};

template <int BOARD_SIZE>
struct TreeNode {
    // 多线程不变部分
    TreeNode* parent_;
    std::bitset<BOARD_SIZE * BOARD_SIZE> availables_;
    std::bitset<BOARD_SIZE * BOARD_SIZE> blacks_;

    // 多线程易变部分
    std::atomic<std::vector<uint64_t>*> children_;
    std::atomic<uint64_t> concurrency_visits_score_;  // 最高1字节代表节点的当前线程并发数量(concurrency), 接着三字节为visits，最低4字节为score
    // float p = 1.0;  // prior probability

    TreeNode() : parent_(nullptr) {
        availables_.flip();
        children_ = new std::vector<uint64_t>();
    }

    TreeNode(TreeNode* parent, int move_index, bool is_last_black) : parent_(parent), availables_(parent->availables_), blacks_(parent->blacks_) {
        availables_.set(move_index, false);
        blacks_.set(move_index, is_last_black);
        children_ = new std::vector<uint64_t>();
    }

    ~TreeNode() {
        // 不管parent的内存
        std::vector<uint64_t>* children = children_;
        for (auto itr = children->begin(); itr != children->end(); ++itr) {
            auto ptr = (TreeNode*)((*itr) & 0x00ffffffffffffff);
            delete ptr;
        }
        delete children;
    }
    
    void MakeVisible() {
        invalidate_cache_region_atomic(this, sizeof(*this));
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

    // 增加1个concurrency
    uint64_t WeakLock() {
        uint64_t old_value, new_value;
        do {
            old_value = concurrency_visits_score_;
            new_value = ((((old_value >> 56) + 1) << 56) | (old_value & 0x00ffffffffffffffuLL));
        } while (!concurrency_visits_score_.compare_exchange_strong(old_value, new_value));
        return new_value;
    }

    auto SelectAndExpand(int task_idx, HPList<std::vector<uint64_t>>& hp_list, std::vector<std::vector<uint64_t>*>& retire_list, std::mt19937& engine, float c_puct, bool is_black) {
        float max_q = -1.0;
        std::vector<int> candidates;

        uint64_t concurrency_visits_score = concurrency_visits_score_;
        uint32_t visits = ((uint32_t)(concurrency_visits_score >> 32) & 0x00ffffffu);
        std::vector<uint64_t>* children = children_;
        for (int i = 0; i < BOARD_SIZE* BOARD_SIZE; ++i) {
            if (!availables_[i]) {
                continue;
            }

            auto itr = find_by_key(*children, (uint8_t)i);
            float q = 0;

            float puct = c_puct * sqrt(visits);
            float virtual_loss = 0.0;

            if (itr != children->end()) {
                auto* child = (TreeNode*)(*itr & 0x00ffffffffffffff);
                uint64_t state = child->concurrency_visits_score_;
                uint8_t child_concurrency = (uint8_t)(state >> 56);
                uint32_t child_visits = ((uint32_t)(state >> 32) & 0x00ffffffu);
                int32_t child_score = (int32_t)(state & 0xffffffff);
                q = (float)child_score / child_visits;
                puct /= (child_visits + 1);
                virtual_loss = -1.0 * child_concurrency;
            }
            auto adjust_q = q + puct + virtual_loss;
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
        std::cerr << format("Task#%d Selected child index: %d\n", task_idx, child_idx);
        for (; ;) {
            HPRecType<std::vector<uint64_t>>* p_rec = hp_list.Acquire();
            std::vector<uint64_t>* old_children;
            do {
                old_children = children_;
                p_rec->hazard_ = old_children;
            } while (children_.load() != old_children);

            std::vector<uint64_t>* children = new std::vector<uint64_t>(*old_children);
            auto itr = find_insertion_point(*children, (uint8_t)child_idx);
            if (itr != children->end() && (*itr >> 56) == child_idx) {
                p_rec->Release();
                return itr;
            }
            auto* entry = new TreeNode<BOARD_SIZE>(this, child_idx, is_black);
            entry->MakeVisible();
            auto elem = ((uint64_t)entry | ((uint64_t)child_idx << 56));
            std::vector<uint64_t>::iterator ret = children->insert(itr, elem);
            MakeChildrenVisible(*children);
            if (children_.compare_exchange_strong(old_children, children)) {
                p_rec->Release();
                hp_list.Retire(retire_list, old_children);
                std::cerr << format("Task#%d Children size after select and expand: %d\n", task_idx, children_.load()->size());
                return ret;
            }
            delete children;
            p_rec->Release();
        }
    }

    void UpdateRecursively(int score) {
        if (parent_ != nullptr) {
            parent_->UpdateRecursively(score);
        }

        uint64_t old_value, new_value;
        do {
            old_value = concurrency_visits_score_;
            uint8_t concurrency = (uint8_t)(old_value >> 56);
            uint32_t visits = ((uint32_t)(old_value >> 32) & 0xffffffffu);
            ++visits;
            int32_t scores = (int32_t)(old_value & 0x00000000ffffffffuLL);
            scores += score;
            new_value = ( ((uint64_t)concurrency << 56) | ((uint64_t)visits << 32) | (uint64_t)scores );
        } while (!concurrency_visits_score_.compare_exchange_weak(old_value, new_value));
    }
};

template <int BOARD_SIZE = 15>
class PureMCTSGame {
public:
    PureMCTSGame(int cores, float c_puct, bool reuse_tree_states = false) : executor_(cores, cores), root_(new TreeNode<BOARD_SIZE>()), last_move_{-1, -1}, c_puct_(c_puct), is_last_black_(false), reuse_tree_states_(reuse_tree_states) {}

    PureMCTSGame(int cores, py::array_t<int32_t> board, Move last_move, float c_puct, bool reuse_tree_states = false) : executor_(cores, cores), root_(new TreeNode<BOARD_SIZE>()), last_move_(last_move), c_puct_(c_puct), reuse_tree_states_(reuse_tree_states) {
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
            std::vector<uint64_t>* root_children = root_->children_;
            auto itr = find_insertion_point(*root_children, (uint8_t)child_idx);
            if (itr == root_children->end() || (*itr >> 56) > (uint64_t)child_idx) {
                 auto* entry = new TreeNode<BOARD_SIZE>(root_, child_idx, !is_last_black_);
                 auto elem = ((uint64_t)entry | ((uint64_t)child_idx << 56));
                 root_children->insert(itr, elem);
            }
            auto it = find_by_key(*root_children, (uint8_t)child_idx);
            new_root = (TreeNode<BOARD_SIZE>*)(*it & 0x00ffffffffffffffuLL);
            root_children->erase(it);
        }
        new_root->parent_ = nullptr;
        delete root_;
        root_ = new_root;
        std::cerr << format("New root children size: %d\n", (int)root_->children_.load()->size());
        last_move_ = { x, y };
        is_last_black_ = !is_last_black_;
   }

    bool IsEnd() const {
        return root_->IsEnd(last_move_, is_last_black_);
    }

    Move SearchBestMove(int simulate_times);
    void Ruleout(HPList<std::vector<uint64_t>>& hp_list, std::atomic<int>& finished_count, int task_idx, std::vector<void*> retire_lists);

private:
    folly::CPUThreadPoolExecutor executor_;
    TreeNode<BOARD_SIZE>* root_;
    Move last_move_;
    float c_puct_;
    bool is_last_black_{false};
    bool reuse_tree_states_;
};

template <int BOARD_SIZE>
Move PureMCTSGame<BOARD_SIZE>::SearchBestMove(int simulate_times) {
    Move ret = { -1, -1 };
    if (root_->IsEnd(last_move_, is_last_black_)) {
        return ret;
    }

    std::atomic<int> finished_count = 0;
    HPList<std::vector<uint64_t>> hp_list;
    std::vector<void*> retire_lists(simulate_times, nullptr);

    MakeAllVisible();
    std::vector<folly::Future<folly::Unit>> futures;
    for (int i = 0; i < simulate_times; ++i) {
        // futures.emplace_back(executor_.add([this, &hp_list, &finished_count, i, &retire_lists]() { this->Ruleout(hp_list, finished_count, i, retire_lists); }));
        folly::Future<folly::Unit> fut = folly::via(&executor_, [this, &hp_list, &finished_count, i, &retire_lists]() { this->Ruleout(hp_list, finished_count, i, retire_lists); });
        futures.emplace_back(std::move(fut));
    }
    // executor_.join();
    // for (auto& fut : futures) {
    for (auto itr = futures.begin(); itr != futures.end(); ++itr) {
        auto& fut = *itr;
        try {
            std::move(fut).get();
            std::cerr << format("Task #%d executed successfully!\n", (int)(itr - futures.begin()));
        } catch (std::exception& e) {
            std::cerr << format("Task #%d failed with exception: %s\n", (int)(itr - futures.begin()), e.what());
        }
    }

    if (__builtin_expect(finished_count != simulate_times, 0)) {  // unlikely
        std::cerr << "Warning: Finished cout is actually " << finished_count << ", expected is" << simulate_times << std::endl;
    }
    hp_list.print_list();
    for (auto itr = retire_lists.begin(); itr != retire_lists.end(); ++itr) {
        if (*itr == nullptr) {
            continue;
        }
        hp_list.TryReclaim(*(std::vector<std::vector<uint64_t>*>*)*itr);
    }

    // find the most visited.
    std::vector<Move> best_actions;
    uint32_t max_visits = 0;
    std::vector<uint64_t>* root_children = root_->children_;
    std::cerr << "Traversing...\n";
    std::cerr << "Children count: " << root_children->size() << std::endl;
    for (auto itr = root_children->begin(); itr != root_children->end(); ++itr) {
        auto* child = (TreeNode<BOARD_SIZE>*)(*itr & 0x00ffffffffffffff);
        auto idx = (int)(*itr >> 56);
        uint64_t concurrency_visits_score = child->concurrency_visits_score_;
        uint32_t child_visits = ((uint32_t)(concurrency_visits_score >> 32) & 0xffffffu);
        if (child_visits > max_visits) {
            max_visits = child_visits;
            best_actions.clear();
        }
        if (child_visits == max_visits) {
            best_actions.emplace_back(Move((idx % BOARD_SIZE), idx / BOARD_SIZE));
        }
    }

    std::cerr << "Initializing random device..." << std::endl;
    std::random_device rd;
    std::cerr << "Initializing engine..." << std::endl;
    std::mt19937 engine(rd());

    std::cerr << "Best action count: " << best_actions.size() << std::endl;
    std::cerr << "Generating random number..." << std::endl;
    std::uniform_int_distribution<int> distribution(0, (int)best_actions.size() - 1);
    return best_actions[distribution(engine)];
}

std::mt19937& get_threadlocal_generator() {
    thread_local std::random_device rd;
/*
    thread_local std::seed_seq seed{
        rd(),
        static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()),
        static_cast<unsigned int>(std::hash<std::thread::id>{}(std::this_thread::get_id())),
        static_cast<unsigned int>(task_idx)  // 使用任务索引增加随机性
    };
*/
    thread_local std::mt19937 engine(rd());
    return engine;
}

template <int BOARD_SIZE>
void PureMCTSGame<BOARD_SIZE>::Ruleout(HPList<std::vector<uint64_t>>& hp_list, std::atomic<int>& finished_count, int task_idx, std::vector<void*> retire_lists) {
    thread_local std::vector<std::vector<uint64_t>*> retire_list;
    auto& engine = get_threadlocal_generator();

    std::cerr << format("Task #%d Registrying retire list...\n", task_idx);
    retire_lists[task_idx] = &retire_list;  // registry

    TreeNode<BOARD_SIZE>* node = root_;
    Move last_move = last_move_;
    bool is_last_black = is_last_black_;
    while (!node->IsEnd(last_move, is_last_black)) {  // step 3: simulate
        auto itr = node->SelectAndExpand(task_idx, hp_list, retire_list, engine, c_puct_, !is_last_black);  // step 1&2: expand->select
        auto idx = (int)(*itr >> 56);
        last_move = { (idx % BOARD_SIZE), (idx / BOARD_SIZE) };
        is_last_black = !is_last_black;
        node = (TreeNode<BOARD_SIZE>*)(*itr & 0x00ffffffffffffffuLL);
        node->WeakLock();
    }
    int score = 0;
    if (node->availables_.count() > 0) {
        score = (is_last_black != is_last_black_) ? 1 : -1;
    }
    node->UpdateRecursively(score);  // step 4: backpropagate
    int count = finished_count.fetch_add(1) + 1;
    if (__builtin_expect((count % 100000) == 0, 0)) {  // unlikely
        std::cout << format("Ruleout count: %d\n", count); 
    }
}
}  // namespace gomoku_ai

PYBIND11_MODULE(gomoku_ai, m) {
    m.doc() = "Pybind11 pure MCTS gomoku plugin";

    py::class_<gomoku_ai::PureMCTSGame<8>>(m, "PureMCTSGame8")
        .def(py::init<int, float, bool>(),
            py::arg("cores"), py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def(py::init<int, py::array_t<int32_t>, gomoku_ai::Move, float, bool>(),
            py::arg("cores"), py::arg("board"), py::arg("last_move"), py::arg("c_puct"), py::arg("reuse_tree_states"),
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
        .def(py::init<int, float, bool>(),
            py::arg("cores"), py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def(py::init<int, py::array_t<int32_t>, gomoku_ai::Move, float, bool>(),
            py::arg("cores"), py::arg("board"), py::arg("last_move"), py::arg("c_puct"), py::arg("reuse_tree_states"),
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
        .def(py::init<int, float, bool>(),
            py::arg("cores"), py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def(py::init<int, py::array_t<int32_t>, gomoku_ai::Move, float, bool>(),
            py::arg("cores"), py::arg("board"), py::arg("last_move"), py::arg("c_puct"), py::arg("reuse_tree_states"),
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
        .def(py::init<int, float, bool>(),
            py::arg("cores"), py::arg("c_puct"), py::arg("reuse_tree_states"),
            "Construct a pure mcts game.\n\n"
            "Args:\n"
            "    c_puct(float): const cofficent of PUCT algorithm.\n"
            "    reuse_tree_states(bool): reuse former MCTS tree states or not\n")
        .def(py::init<int, py::array_t<int32_t>, gomoku_ai::Move, float, bool>(),
            py::arg("cores"), py::arg("board"), py::arg("last_move"), py::arg("c_puct"), py::arg("reuse_tree_states"),
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


