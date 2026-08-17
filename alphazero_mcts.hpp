#ifndef ALPHAZERO_MCTS_HPP_
#define ALPHAZERO_MCTS_HPP_

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <map>
#include <shared_mutex>
#include <vector>
#include <random>
#include <type_traits>
#include <sys/stat.h>

#include <torch/script.h>
#include <ATen/Parallel.h>

// Deliberately do NOT include omp.h to avoid pulling in OpenMP compile
// flags; libgomp is already loaded into the process by torch, so linking
// with -lgomp is enough to resolve the symbol (see setup.py).
// Note: do NOT call mkl_set_num_threads_local here -- calling it first on a
// worker thread whose MKL TLS was never initialized segfaults (observed);
// if ever needed, it must be called after the first MKL op on that thread.
extern "C" void omp_set_num_threads(int);
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/Unit.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace gomoku_ai {
void softmax_inplace(std::vector<double>& input) {
    double max_val = *std::max_element(input.begin(), input.end());
    double sum = 0.0;

    // Compute exponentials and their sum, with the max subtracted for
    // numerical stability.
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::exp(input[i] - max_val);
        sum += input[i];
    }

    // Normalize
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] /= sum;
    }
}

struct High8Compare {
    bool operator()(uint64_t a, uint64_t b) const noexcept {
        return (a & 0xFF00000000000000) < (b & 0xFF00000000000000);
    }
};

// find the first >= target iterator.
inline auto find_insertion_point(std::vector<uint64_t>& vec, uint8_t key) {
    uint64_t key_pattern = static_cast<uint64_t>(key) << 56;
    return std::lower_bound(vec.begin(), vec.end(), key_pattern, High8Compare());
}

inline bool try_insert_sorted(std::vector<uint64_t>& vec, uint64_t value) {
    uint8_t key = value >> 56;
    auto it = find_insertion_point(vec, key);
    
    if ((it != vec.end() && (*it >> 56) == key)) {
        return false;
    }
    
    vec.insert(it, value);
    return true;
}

inline auto find_by_key(const std::vector<uint64_t>& vec, uint8_t key) {
    uint64_t key_pattern = static_cast<uint64_t>(key) << 56;
    auto it = std::lower_bound(vec.begin(), vec.end(), key_pattern, High8Compare());
    
    if (it != vec.end() && (*it >> 56) == key) {
        return it;
    }
    return vec.end();
}

inline void invalidate_cache_region_atomic(void* data_ptr, size_t size) {
    auto* masked_data_ptr = (void*)((uint64_t)data_ptr & 0xffffffffffffff40uLL);
    auto adjusted_size = size + ((uint8_t*)data_ptr - (uint8_t*)masked_data_ptr);
    auto* ptr = static_cast<std::atomic<uintptr_t>*>(masked_data_ptr);
   
    for (size_t i = 0; i < adjusted_size; ++i, ptr += 64 / sizeof(uintptr_t)) { 
        (*ptr).fetch_or(0, std::memory_order_acq_rel);
    }
    
    // std::atomic_thread_fence(std::memory_order_seq_cst);
}

void MakeChildrenVisible(std::vector<uint64_t>& children) {
    invalidate_cache_region_atomic((void*)children.data(), children.size() * sizeof(uint64_t));
    invalidate_cache_region_atomic(&children, sizeof(children));
}

inline void MakeAllVisible() {
    std::mutex mut;
    std::lock_guard lock(mut);
}

typedef std::pair<int, int> Move;  // first->x,second->y
inline std::string format(const char* fmt, ...) {
    va_list args;
    
    va_start(args, fmt);
    int length = vsnprintf(nullptr, 0, fmt, args); // per C++11, if buf is nullptr and size is 0, returns the number of bytes needed (excluding the null terminator)
    va_end(args);
    
    if (length <= 0) {
        return ""; // formatting error: return an empty string
    }
    
    size_t buf_size = length + 1;
    std::vector<char> buf(buf_size);
    
    va_start(args, fmt);
    vsnprintf(buf.data(), buf_size, fmt, args); // use the vector's data() member to get the raw pointer
    va_end(args);
    
    return std::string(buf.data());
}

// Hazard pointer
template <class TYPE>
struct HPRecType {
   std::atomic<HPRecType<TYPE>*> next_ = nullptr;
   std::atomic<bool> active_ = false;
   std::atomic<TYPE*> hazard_ = nullptr;  // NOTE(junhaozhang): not owned!

   void Release() {
        hazard_ = nullptr;
        active_ = false;
   }
};

// Hazard pointer list
template <class TYPE, size_t R = 1000>
class HPList {
public:
    HPList(size_t prealloc_size) {
        head_ = nullptr;

        while (prealloc_size) {
            HPRecType<TYPE>* p = new HPRecType<TYPE>();
            p->next_ = head_.load();
            head_ = p;
            --prealloc_size;
        }           
    }

    HPRecType<TYPE>* Acquire() {
       HPRecType<TYPE>* p = head_;
        for (; p; p = p->next_.load()) {
            // NOTE(junhaozhang): expected must be reset on every iteration!
            // A failed CAS overwrites expected with the current value (true);
            // without resetting, the next CAS would compare true against
            // someone else's active rec and "succeed" -- two threads would
            // share one HP rec, hazard protection breaks, a retired children
            // vector gets reclaimed early, and the wild-pointer read recurses
            // forever inside uniform_int_distribution(0,-1) until the stack
            // blows up (confirmed by a core dump from an 8000-game training
            // run).
            bool expected_false = false;
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
            HPRecType<TYPE>* next = p->next_.load();
            delete p;
            p = next;
        }
        head_ = nullptr;
    }

    void Retire(std::vector<TYPE*>& retire_list, TYPE* p, int task_idx = -1) {
        retire_list.push_back(p);
        if (retire_list.size() < R) {
            return;
        }
        TryReclaim(retire_list, task_idx);
    }

    void TryReclaim(std::vector<TYPE*>& retire_list, int task_idx = -1) {
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
                // std::cerr << format("Task #%d Reclaim %p\n", task_idx, *itr);
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
        // std::cerr << "HP_LIST: ";
        while (p) {
            // std::cerr << p->active_ << "|" << p->hazard_ << " -> ";
            p = p->next_;
        }
        // std::cerr << std::endl;
    }

private:
    std::atomic<HPRecType<TYPE>*> head_ = nullptr;
};

template <class TYPE>
class MTRetireLists {
public:
    void InheritThreadLocalRetireList(std::vector<TYPE*>& retire_list) {
        std::thread::id this_id = std::this_thread::get_id();
        {
        std::shared_lock rlock(mutex_);
        auto itr = lists_.find(this_id);
        if (itr == lists_.end() || itr->second.empty()) {
            return;
        }
        }

        std::unique_lock wlock(mutex_);
        retire_list = std::move(lists_[this_id]);
    }

    void UpdateThreadLocalRetireList(std::vector<TYPE*>&& retire_list) {
        if (retire_list.empty()) {
            return;
        }

        std::thread::id this_id = std::this_thread::get_id();
        std::unique_lock wlock(mutex_);
        lists_[this_id] = retire_list;
    }

    std::map<std::thread::id, std::vector<TYPE*>>& GetLists() {
        std::unique_lock wlock(mutex_);
        return lists_;
    }

private:
    mutable std::shared_mutex mutex_;
    std::map<std::thread::id, std::vector<TYPE*>> lists_;
};

class ThreadLocalModels {
public:
    ThreadLocalModels(const char* model_path) : model_path_(model_path) {}

    // Global model cache reused across moves: when the same model_path is
    // unchanged (mtime+size), reuse the thread-local models; when the file
    // changes (e.g. self-play rewrites the .pt after every game), rebuild the
    // whole thing. Otherwise every move would make each of the N worker
    // threads torch::jit::load once (one load + first forward takes
    // 27~57ms; x N threads x ~50 moves ~ tens of seconds per game).
    // NOTE(junhaozhang): the registry is a deliberately leaked heap object so
    // that torch modules are never destructed at process exit, avoiding an
    // exit crash when the torch runtime is torn down first; mid-process
    // replacement happens while torch is alive and is safe.
    static ThreadLocalModels& Get(const char* model_path) {
        struct stat st;
        uint64_t mtime_ns = 0, fsize = 0;
        if (::stat(model_path, &st) == 0) {
            mtime_ns = (uint64_t)st.st_mtim.tv_sec * 1000000000ull + (uint64_t)st.st_mtim.tv_nsec;
            fsize = (uint64_t)st.st_size;
        }
        std::lock_guard<std::mutex> g(RegistryMutex());
        auto& entry = Registry()[model_path];
        if (!entry.models || entry.mtime_ns != mtime_ns || entry.fsize != fsize) {
            entry.models.reset(new ThreadLocalModels(model_path));
            entry.mtime_ns = mtime_ns;
            entry.fsize = fsize;
        }
        return *entry.models;
    }

    // NOTE(junhaozhang): the c10::Error thrown by torch::jit::load cannot
    // propagate safely out of a folly worker (unwinding aborts the whole
    // process, observed in practice), so here we do a limited backoff-retry
    // for "model file is being replaced / temporarily unreadable"; the
    // exception is rethrown only after all retries are exhausted.
    static torch::jit::script::Module LoadWithRetry(const std::string& path, int max_retries = 100) {
        for (int attempt = 0; ; ++attempt) {
            try {
                return torch::jit::load(path);
            } catch (const std::exception& e) {
                if (attempt >= max_retries) {
                    std::cerr << format("FATAL: load model %s failed after %d retries: %s\n",
                                        path.c_str(), max_retries, e.what());
                    throw;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }

    torch::jit::script::Module& GetThreadLocalModel() {
        std::thread::id this_id = std::this_thread::get_id();
        {
        std::shared_lock rlock(mutex_);
        auto itr = models_.find(this_id);
        if (itr != models_.end()) {
            return itr->second;
        }
        }

        torch::jit::script::Module module = LoadWithRetry(model_path_);
        // std::cerr << format("Thread #%llu Model load successfully!\n", *(uint64_t*)&this_id);
        std::unique_lock wlock(mutex_);
        models_[this_id] = std::move(module);
        return models_[this_id];
    }

private:
    struct Entry {
        std::unique_ptr<ThreadLocalModels> models;
        uint64_t mtime_ns = 0;
        uint64_t fsize = 0;
    };
    static std::map<std::string, Entry>& Registry() {
        static auto* registry = new std::map<std::string, Entry>();  // deliberately leaked; see the comment on Get
        return *registry;
    }
    static std::mutex& RegistryMutex() {
        static auto* m = new std::mutex();
        return *m;
    }
    mutable std::shared_mutex mutex_;
    std::map<std::thread::id, torch::jit::script::Module>  models_;
    std::string model_path_;
};

template <int BOARD_SIZE, bool WITH_PRIOR_P, bool WITH_VIRTUAL_LOSS = false>
struct TreeNode {
    // Parts that are immutable under multithreading
    TreeNode* parent_;
    std::bitset<BOARD_SIZE * BOARD_SIZE> availables_;
    std::bitset<BOARD_SIZE * BOARD_SIZE> blacks_;

    // Parts that are mutable under multithreading
    std::atomic<uint64_t> children_;  // the top two bytes hold a counter to avoid the ABA problem
    std::atomic<uint64_t> concurrency_visits_score_ = 0;  // top 1 byte: number of threads currently inside the node (concurrency), next 3 bytes: visits, lowest 4 bytes: score
    std::conditional_t<WITH_PRIOR_P == true, float, std::monostate> p_;  // prior probability

    template <bool T = WITH_PRIOR_P, typename std::enable_if_t<!T, bool> = true>
    TreeNode() : parent_(nullptr) {
        availables_.flip();
        children_ = (uint64_t)new std::vector<uint64_t>();
    }

    template <bool T = WITH_PRIOR_P, typename std::enable_if_t<T, bool> = true> 
    TreeNode(float p = 1.0) : parent_(nullptr), p_(p) {
        availables_.flip();
        children_ = (uint64_t)new std::vector<uint64_t>();
    }

    template <bool T = WITH_PRIOR_P, typename std::enable_if_t<!T, bool> = true>
    TreeNode(TreeNode* parent, int move_index, bool is_last_black) : parent_(parent), availables_(parent->availables_), blacks_(parent->blacks_) {
        availables_.set(move_index, false);
        blacks_.set(move_index, is_last_black);
        children_ = (uint64_t)new std::vector<uint64_t>();
    }

    template <bool T = WITH_PRIOR_P, typename std::enable_if_t<T, bool> = true>
    TreeNode(TreeNode* parent, int move_index, bool is_last_black, float p = 1.0) : parent_(parent), availables_(parent->availables_), blacks_(parent->blacks_), p_(p) {
        availables_.set(move_index, false);
        blacks_.set(move_index, is_last_black);
        children_ = (uint64_t)new std::vector<uint64_t>();
    }

    ~TreeNode() {
        // parent's memory is not managed here
        std::vector<uint64_t>* children = (std::vector<uint64_t>*)(children_ & 0x0000ffffffffffff);
        // std::cerr << format("Reclaim %zu children entries!\n", children->size());
        for (auto itr = children->begin(); itr != children->end(); ++itr) {
            auto ptr = (TreeNode*)((*itr) & 0x00ffffffffffffff);
            delete ptr;
        }
        delete (std::vector<uint64_t>*)(children_ & 0x0000ffffffffffffuLL);
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

        // Horizontal
        int diff_begin = 0;
        while (last_move.first - diff_begin >= 0 && solid[idx - diff_begin]) { ++diff_begin; }
        int diff_end = 0;
        while (last_move.first + diff_end < BOARD_SIZE && solid[idx + diff_end]) { ++diff_end; }
        if (diff_end + diff_begin - 1 >= 5) {
            return true;
        }

        // Vertical
        diff_begin = 0;
        while (last_move.second - diff_begin >= 0 && solid[idx - diff_begin * BOARD_SIZE]) { ++diff_begin; }
        diff_end = 0;
        while (last_move.second + diff_end < BOARD_SIZE && solid[idx + diff_end * BOARD_SIZE]) { ++diff_end; }
        if (diff_end + diff_begin - 1 >= 5) {
            return true;
        }

        // Diagonal '\'
        diff_begin = 0;
        while (last_move.first - diff_begin >= 0 && last_move.second - diff_begin >= 0 && solid[idx - diff_begin * (BOARD_SIZE + 1)]) { ++diff_begin; }
        diff_end = 0;
        while (last_move.first + diff_end < BOARD_SIZE && last_move.second + diff_end < BOARD_SIZE && solid[idx + diff_end * (BOARD_SIZE + 1)]) { ++diff_end; }
        if (diff_end + diff_begin - 1 >= 5) {
            return true;
        }

        // Diagonal '/'
        diff_begin = 0;
        while (last_move.first - diff_begin >= 0 && last_move.second + diff_begin < BOARD_SIZE && solid[idx + diff_begin * (BOARD_SIZE - 1)]) { ++diff_begin; }
        diff_end = 0;
        while (last_move.first + diff_end < BOARD_SIZE && last_move.second - diff_end >= 0 && solid[idx - diff_end * (BOARD_SIZE - 1)]) { ++diff_end; }
        if (diff_end + diff_begin - 1 >= 5) {
            return true;
        }

        return false;
    }        

    // Increment concurrency by 1
    uint64_t WeakLock() {
        uint64_t old_value, new_value;
        do {
            old_value = concurrency_visits_score_;
            new_value = ((((old_value >> 56) + 1) << 56) | (old_value & 0x00ffffffffffffffuLL));
        } while (!concurrency_visits_score_.compare_exchange_strong(old_value, new_value));
        // std::cerr << format("WEAKLOCK: %llx -> %llx\n", old_value, new_value);
        return new_value;
    }

    void GenModelInputTensor(torch::Tensor& input_tensor, Move& last_move, bool is_last_black) {
        // NOTE(jhzhang03): assumes input_tensor is all zeros
        // channel 1: current player's stones
        // channel 2: opponent's stones
        // channel 3: last move position
        // channel 4: side to move (1.0 if black is next)
        auto accessor = input_tensor.accessor<float, 4>();
        if (!is_last_black) {  // black plays next
            for (int x = 0; x < BOARD_SIZE; ++x) {
                for (int y = 0; y <BOARD_SIZE; ++y) {
                    int idx = y * BOARD_SIZE + x;
                    if (blacks_[idx]) {
                        accessor[0][0][y][x] = 1.0;
                    } else if (!availables_[idx]) {
                        accessor[0][1][y][x] = 1.0;
                    }
                    // NOTE(junhaozhang): channel 4 means "black moves next"
                    // and must be filled with 1.0 across the whole plane, not
                    // just the empty points, otherwise it mismatches the
                    // training input produced by game.py on the Python side.
                    accessor[0][3][y][x] = 1.0;
                }
            }
        } else {
            for (int x = 0; x < BOARD_SIZE; ++x) {
                for (int y = 0; y < BOARD_SIZE; ++y) {
                    int idx = y * BOARD_SIZE+x;
                    if (blacks_[idx]) {
                        accessor[0][1][y][x] = 1.0;
                    } else if (!availables_[idx]) {
                        accessor[0][0][y][x] = 1.0;
                    }
                }
            }
        }
        if (last_move.first >= 0 && last_move.second >= 0) {
            accessor[0][2][last_move.second][last_move.first] = 1.0;
        }
    }

    template <bool T = WITH_PRIOR_P, typename std::enable_if_t<T, bool> = true>
    void Expand(const std::map<int, float>& act_probs, bool is_black) {
        uint64_t old_children_with_count;
        std::vector<uint64_t>* old_children;
        do {
            old_children_with_count = children_.load();
            old_children = (std::vector<uint64_t>*)(old_children_with_count & 0x0000ffffffffffffuLL);
            if (old_children != nullptr && !old_children->empty()) {
                return;
            }
        } while (children_.load() != old_children_with_count);

        auto* children = new std::vector<uint64_t>();
        for (auto itr = act_probs.begin(); itr != act_probs.end(); ++itr) {
            int child_idx = itr->first;
            if (!availables_[child_idx]) {
                continue;
            }
            TreeNode<BOARD_SIZE, WITH_PRIOR_P> * entry;
            if constexpr (WITH_PRIOR_P) {
                entry = new TreeNode<BOARD_SIZE, WITH_PRIOR_P>(this, child_idx, is_black, itr->second);
            } else {
                entry = new TreeNode<BOARD_SIZE, WITH_PRIOR_P>(this, child_idx, is_black);
            }
            entry->MakeVisible();
            // std::cerr << "CHILD_IDX: " << itr->first << ", ENTRY: " << entry << ", Prior P=" << itr->second << std::endl;
            auto elem = ((uint64_t)entry | ((uint64_t)child_idx << 56));
            children->insert(children->end(), elem);
        }
        MakeChildrenVisible(*children);
        uint64_t children_with_count = ( ((uint64_t)((uint16_t)(old_children_with_count >> 48) + 1) << 48) | (uint64_t)children );
        if (!children_.compare_exchange_strong(old_children_with_count, children_with_count)) {
            for (auto itr = children->begin(); itr != children->end(); ++itr) {
                auto* entry = (TreeNode<BOARD_SIZE, WITH_PRIOR_P>*)(*itr & 0x00ffffffffffffffuLL);
                delete entry;
            }
            delete children;
        }
    }

    uint64_t SelectAndExpand(int task_idx, HPList<std::vector<uint64_t>>& hp_list, std::vector<std::vector<uint64_t>*>& retire_list, std::mt19937& engine, float c_puct, bool is_black) {
        float max_q = -1.0;
        std::vector<int> candidates;

        uint64_t concurrency_visits_score = concurrency_visits_score_;
        uint32_t visits = ((uint32_t)(concurrency_visits_score >> 32) & 0x00ffffffu);

        HPRecType<std::vector<uint64_t>>* p_rec = hp_list.Acquire();
        uint64_t children_with_count;
        std::vector<uint64_t>* children;
        do {
            children_with_count = children_.load();
            children = (std::vector<uint64_t>*)(children_with_count & 0x0000ffffffffffffuLL);
            p_rec->hazard_ = children;
        } while (children_.load() != children_with_count);

        for (int i = 0; i < BOARD_SIZE* BOARD_SIZE; ++i) {
            // std::cerr << "BITCOUNT: " << availables_.count() << std::endl;
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
                // std::cerr << format("P_: %.2f\n", child->p_);
                if constexpr (WITH_PRIOR_P) {
                    puct *= child->p_;
                    float child_score;
                    std::memcpy(&child_score, &state, sizeof(child_score));
                    if (child_visits != 0) {
                        q = child_score / child_visits;
                    }
                } else {
                    int32_t child_score = (int32_t)(state & 0xffffffffuLL);
                    q = (float)child_score / child_visits;
                }
                if constexpr (!WITH_VIRTUAL_LOSS) {
                    puct /= (child_visits + child_concurrency + 1);
                } else {
                    puct /= (child_visits + 1);
                    virtual_loss = -1.0 * child_concurrency;
                }
            }
            auto adjust_q = q + puct + virtual_loss;
            // std::cerr << format("ADJUST_Q: %.2f\n", adjust_q);
            if (adjust_q > max_q) {
                candidates.clear();
                max_q = adjust_q;
            }
            if (adjust_q >= max_q-1e-5) {
                candidates.emplace_back(i);
            }
        }

        p_rec->Release();
        // std::cerr << format("Candidates size: %llu\n", candidates.size());
        std::uniform_int_distribution<int> distribution(0, (int)candidates.size() - 1);
        int child_idx = candidates[distribution(engine)];
        // std::cerr << format("Task#%d Selected child index: %d\n", task_idx, child_idx);
        for (; ;) {
            HPRecType<std::vector<uint64_t>>* p_rec = hp_list.Acquire();
            uint64_t old_children_with_count;
            std::vector<uint64_t>* old_children;
            do {
                old_children_with_count = children_.load();
                old_children = (std::vector<uint64_t>*)(old_children_with_count & 0x0000ffffffffffffuLL);
                p_rec->hazard_ = old_children;
            } while (children_.load() != old_children_with_count);

            // std::cerr << format("Task #%d Visiting %p, hp rec=%p\n", task_idx, old_children, p_rec);
            auto itr = find_insertion_point(*old_children, (uint8_t)child_idx);
            if (itr != old_children->end() && (*itr >> 56) == child_idx) {
                p_rec->Release();
                return *itr;
            }
            // std::cerr << format("SIZE: %llu\n", itr - old_children->begin()); 
            std::vector<uint64_t>* children = new std::vector<uint64_t>(old_children->begin(), itr);
            auto* entry = new TreeNode<BOARD_SIZE, WITH_PRIOR_P>(this, child_idx, is_black);
            entry->MakeVisible();
            auto elem = ((uint64_t)entry | ((uint64_t)child_idx << 56));
            // std::cerr << format("ELEM: %llx\n", elem);
            children->insert(children->end(), elem);
            children->insert(children->end(), itr, old_children->end());
            MakeChildrenVisible(*children);
            uint64_t children_with_count = ( ((uint64_t)((uint16_t)(old_children_with_count >> 48) + 1) << 48) | (uint64_t)children );
            // std::cerr << format("Task #%d %llx -> %llx\n", task_idx, old_children_with_count, children_with_count);
            if (children_.compare_exchange_strong(old_children_with_count, children_with_count)) {
                // std::cerr << format("Task #%d: Already allocate %p\n", task_idx, children);
                p_rec->Release();
                hp_list.Retire(retire_list, old_children, task_idx);
                // std::cerr << format("Task#%d Children size after select and expand: %d,itr=%llx\n", task_idx, children_.load()->size(), elem);
                return elem;
            }
            delete entry;
            delete children;
            p_rec->Release();
        }
    }

    template <bool T = WITH_PRIOR_P, typename std::enable_if_t<!T, bool> = true>
    void UpdateRecursively(int score) {
        if (parent_ != nullptr) {
            parent_->UpdateRecursively(-score);
        }

        uint64_t old_value, new_value;
        do {
            old_value = concurrency_visits_score_;
            uint8_t concurrency = (uint8_t)(old_value >> 56);
            --concurrency;
            uint32_t visits = ((uint32_t)(old_value >> 32) & 0x00ffffffu);
            ++visits;
            int32_t scores = (int32_t)(old_value & 0x00000000ffffffffuLL);
            scores += score;
            new_value = ( ((uint64_t)concurrency << 56) | ((uint64_t)visits << 32) | (uint64_t)(uint32_t)scores );
        } while (!concurrency_visits_score_.compare_exchange_strong(old_value, new_value));
        // std::cerr << format("Update State: %llx -> %llx\n", old_value, new_value);
    }

    template <bool T = WITH_PRIOR_P, typename std::enable_if_t<T, bool> = true>
    void UpdateRecursively(float score) {
        if (parent_ != nullptr) {
            parent_->UpdateRecursively(-score);
        }

        uint64_t old_value, new_value;
        do {
            old_value = concurrency_visits_score_;
            uint8_t concurrency = (uint8_t)(old_value >> 56);
            --concurrency;
            uint32_t visits = ((uint32_t)(old_value >> 32) & 0x00ffffffu);
            ++visits;
            float scores;
            std::memcpy(&scores, &old_value, sizeof(scores));
            scores += score;
            // NOTE(junhaozhang): memcpy only writes the low 32 bits; the high
            // 32 bits must be zeroed before the OR, otherwise they contain
            // uninitialized data / residue from the previous CAS round and
            // would corrupt visits and concurrency.
            new_value = 0;
            std::memcpy(&new_value, &scores, sizeof(scores));
            new_value |= ((uint64_t)concurrency << 56);
            new_value |= ((uint64_t)visits << 32);
        } while (!concurrency_visits_score_.compare_exchange_strong(old_value, new_value));
    }
};

template <int BOARD_SIZE, bool WITH_MODEL, bool WITH_VIRTUAL_LOSS = false>
class GomokuMCTSFramework {
public:
    GomokuMCTSFramework(int cores, float c_puct, bool reuse_tree_states = false) : executor_(cores, cores), root_(new TreeNode<BOARD_SIZE, WITH_MODEL>()), last_move_{-1, -1}, c_puct_(c_puct), is_last_black_(false), reuse_tree_states_(reuse_tree_states) {}

    GomokuMCTSFramework(int cores, py::array_t<int32_t> board, Move last_move, float c_puct, bool reuse_tree_states = false) : executor_(cores, cores), root_(new TreeNode<BOARD_SIZE, WITH_MODEL>()), last_move_(last_move), c_puct_(c_puct), reuse_tree_states_(reuse_tree_states) {
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
                throw std::runtime_error(format("NO last action but there are %d pieces on board!", BOARD_SIZE * BOARD_SIZE - root_->availables_.count()));
            }
            is_last_black_ = false;  // black is strictly required to move first
        } else {
            int index = last_move_.second * BOARD_SIZE + last_move_.first;
            if (root_->availables_[index]) {
                throw std::runtime_error(format("The move(%d,%d) has not been made!", last_move.first, last_move.second));
            }
            is_last_black_ = root_->blacks_[index];
        }
    }

    GomokuMCTSFramework(int cores, std::vector<std::vector<int>>& board, Move last_move, float c_puct, bool reuse_tree_states = false) : executor_(cores, cores), root_(new TreeNode<BOARD_SIZE, WITH_MODEL>()), last_move_(last_move), c_puct_(c_puct), reuse_tree_states_(reuse_tree_states) {
        if (last_move_.first >= BOARD_SIZE || last_move_.second >= BOARD_SIZE) {
            throw std::runtime_error(format("Last move(%d,%d) is out of board(board size:%d)!", last_move.first, last_move.second, BOARD_SIZE));
        }

        for (int y = 0; y < BOARD_SIZE; ++y) {
           for (int x = 0; x < BOARD_SIZE; ++x) {
               int index = y * BOARD_SIZE + x;
               if (board[x][y] == 1) {
                   root_->availables_.set(index, false);
                   root_->blacks_.set(index, true);
               } else if (board[x][y] == -1) {
                   root_->availables_.set(index, false);
                   root_->blacks_.set(index, false);
               } else if (board[x][y] == 0) {
                   root_->availables_.set(index, true);
                   root_->blacks_.set(index, false);
               } else {
                   throw std::runtime_error(format("Board(%d,%d) = %d is not in [-1, 0, 1]!", x, y, board[y][x]));
               }
           }
        }
        if (last_move_.first < 0 || last_move_.second < 0) {
            if (root_->availables_.count() != BOARD_SIZE * BOARD_SIZE) {
                throw std::runtime_error(format("NO last action but there are %d pieces on board!", BOARD_SIZE * BOARD_SIZE - root_->availables_.count()));
            }
            is_last_black_ = false;  // black is strictly required to move first
        } else {
            int index = last_move_.second * BOARD_SIZE + last_move_.first;
            if (root_->availables_[index]) {
                throw std::runtime_error(format("The move(%d,%d) has not been made!", last_move.first, last_move.second));
            }
            is_last_black_ = root_->blacks_[index];
        }
    }

    ~GomokuMCTSFramework() {
        // std::cerr << "Reclaim root...\n";
        delete root_;
    }

    // Clear the search tree back to the empty-board opening state
    // (equivalent to rebuilding the Framework), but keep the thread pool:
    // thread ids stay the same, so the models cached by thread id in
    // ThreadLocalModels can be reused across games.
    // NOTE(junhaozhang): if a new Framework were created per game, the new
    // thread pool's new thread ids would make ThreadLocalModels::models_
    // permanently cache `cores` extra torch modules per game (memory leaks
    // with the number of games).
    void Reset() {
        delete root_;
        root_ = new TreeNode<BOARD_SIZE, WITH_MODEL>();
        last_move_ = { -1, -1 };
        is_last_black_ = false;
    }

    bool StateEquals(std::vector<std::vector<int>>& board, bool is_last_black) const {
        if (is_last_black != is_last_black_) {
std::cerr << "Last black not equal!" << std::endl;
            return false;
        }

        auto availables = root_->availables_;
        auto blacks = root_->blacks_;
        for (int y = 0; y < BOARD_SIZE; ++y) {
           for (int x = 0; x < BOARD_SIZE; ++x) {
               int index = y * BOARD_SIZE + x;
               if (board[x][y] == 1) {
                   availables.set(index, false);
                   blacks.set(index, true);
               } else if (board[x][y] == -1) {
                   availables.set(index, false);
                   blacks.set(index, false);
               } else if (board[x][y] == 0) {
                   availables.set(index, true);
                   blacks.set(index, false);
               } else {
                   throw std::runtime_error(format("Board(%d,%d) = %d is not in [-1, 0, 1]!", x, y, board[y][x]));
               }
           }
        }
// std::cerr << "Availables Equals: " << availables == root_->availables_ << std::endl;
// std::cerr << "Blacks Equals: " << blacks == root_->blacks_ << std::endl;
        return (availables == root_->availables_ && blacks == root_->blacks_); 
    }

    bool NpStateEquals(py::array_t<int32_t> np_board, bool is_last_black) const {
        if (is_last_black != is_last_black_) {
// std::cerr << "Last black not equal!" << std::endl;
            return false;
        }

        py::buffer_info buffer_info = np_board.request();
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
// std::cerr << "Availables Equals: " << (availables == root_->availables_) << std::endl;
// std::cerr << "Available count: " << availables.count() << ", " << root_->availables_.count() << std::endl;
// std::cerr << "Blacks Equals: " << (blacks == root_->blacks_) << std::endl;
        return (availables == root_->availables_ && blacks == root_->blacks_); 
    }

    int AvailableCount() const {
        return (int)root_->availables_.count();
    }

    void Play(int x, int y) {
        TreeNode<BOARD_SIZE, WITH_MODEL>* new_root = nullptr;
        int child_idx = y * BOARD_SIZE + x;
        if (!root_->availables_[child_idx]) {
            throw std::runtime_error(format("(%d,%d) is not available!", x, y));
        }
        if (!reuse_tree_states_) {
            new_root = new TreeNode<BOARD_SIZE, WITH_MODEL>(root_, child_idx, !is_last_black_);
        } else {
            std::vector<uint64_t>* root_children = (std::vector<uint64_t>*)(root_->children_ & 0x0000ffffffffffffuLL);
            auto itr = find_insertion_point(*root_children, (uint8_t)child_idx);
            if (itr == root_children->end() || (*itr >> 56) > (uint64_t)child_idx) {
                 auto* entry = new TreeNode<BOARD_SIZE, WITH_MODEL>(root_, child_idx, !is_last_black_);
                 auto elem = ((uint64_t)entry | ((uint64_t)child_idx << 56));
                 root_children->insert(itr, elem);
            }
            auto it = find_by_key(*root_children, (uint8_t)child_idx);
            new_root = (TreeNode<BOARD_SIZE, WITH_MODEL>*)(*it & 0x00ffffffffffffffuLL);
            root_children->erase(it);
        }
        new_root->parent_ = nullptr;

        std::vector<folly::Future<folly::Unit>> futures;
        std::vector<uint64_t>* root_children = (std::vector<uint64_t>*)(root_->children_ & 0x0000ffffffffffffuLL);
        for (auto itr = root_children->begin(); itr != root_children->end(); ++itr) {
            auto ptr = (TreeNode<BOARD_SIZE, WITH_MODEL>*)((*itr) & 0x00ffffffffffffff);
            auto fut = folly::via(&executor_, [ptr]() { delete ptr; });
            futures.emplace_back(std::move(fut));
        }
        delete root_children;
        ::operator delete(root_);
        root_ = new_root;
        // std::cerr << format("New root children size: %d\n", (int)root_->children_.load()->size());
        last_move_ = { x, y };
        is_last_black_ = !is_last_black_;
        for (auto& fut : futures) {
            std::move(fut).get();
        }
   }

    bool IsEnd() const {
        return root_->IsEnd(last_move_, is_last_black_);
    }

    // Return all legal moves and their probabilities
    template <bool W = WITH_MODEL>
    std::enable_if_t<W, std::pair<std::vector<int>, std::vector<double>>> SearchBestMoveWithModel(int simulate_times, const char* model_path, float temperature) {
        std::vector<int> sensible_moves;
        std::vector<double> sensible_probs;
        // std::cerr << "Judging isend...\n";
        if (root_->IsEnd(last_move_, is_last_black_)) {
            return { sensible_moves, sensible_probs };
        }

        std::atomic<int> finished_count = 0;
        HPList<std::vector<uint64_t>> hp_list(100);
        MTRetireLists<std::vector<uint64_t>> retire_lists;

        auto& models = ThreadLocalModels::Get(model_path);  // reused across moves; reload only when the file changes
        MakeAllVisible();
        std::vector<folly::Future<folly::Unit>> futures;
        for (int i = 0; i < simulate_times; ++i) {
            folly::Future<folly::Unit> fut = folly::via(&executor_, [this, &hp_list, &models, &finished_count, i, &retire_lists]() { this->RolloutWithModel(models, hp_list, finished_count, i, retire_lists); });
            futures.emplace_back(std::move(fut));
        }
        // std::cerr << format("All submitted!\n");
        for (auto itr = futures.begin(); itr != futures.end(); ++itr) {
            auto& fut = *itr;
            // std::cerr << format("Getting task #{}", itr - futures.begin());
            std::move(fut).get();
        }

        size_t total_rlist_size = 0;
        auto& rl_data = retire_lists.GetLists();
        for (auto itr = rl_data.begin(); itr != rl_data.end(); ++itr) {
            total_rlist_size += itr->second.size();
            hp_list.TryReclaim(itr->second);
        }
        // std::cerr << "Total retire list size: " << total_rlist_size << std::endl;

        std::vector<uint64_t>* root_children = (std::vector<uint64_t>*)(root_->children_ & 0x0000ffffffffffffuLL);
        // std::cerr << "Root children size: " << root_children->size() << std::endl;
        for (auto itr = root_children->begin(); itr != root_children->end(); ++itr) {
            auto idx = (int)(*itr >> 56);
            if (!root_->availables_[idx]) {
                continue;
            }
            auto* child = (TreeNode<BOARD_SIZE, WITH_MODEL>*)(*itr & 0x00ffffffffffffff);
            uint64_t concurrency_visits_score = child->concurrency_visits_score_;
            uint32_t child_visits = ((uint32_t)(concurrency_visits_score >> 32) & 0xffffffu);
            sensible_probs.push_back(1.0 / temperature * std::log(child_visits + 1.0e-10));
            // std::cerr << "Idx: " << idx << ", visits: " << child_visits << std::endl;
            sensible_moves.push_back(idx);
        }
        softmax_inplace(sensible_probs);
        // std::cerr << "Moves size: " << sensible_moves.size() << std::endl;
        // std::cerr << "Probs size: " << sensible_probs.size() << std::endl;
        return { sensible_moves, sensible_probs };
    }

    template <bool W = WITH_MODEL>
    std::enable_if_t<!W, Move> SearchBestMove(int simulate_times) {
        Move ret = { -1, -1 };
        if (root_->IsEnd(last_move_, is_last_black_)) {
            return ret;
        }
    
        std::atomic<int> finished_count = 0;
        HPList<std::vector<uint64_t>> hp_list(100);
        MTRetireLists<std::vector<uint64_t>> retire_lists;
 
        MakeAllVisible();
        std::vector<folly::Future<folly::Unit>> futures;
        for (int i = 0; i < simulate_times; ++i) {
            folly::Future<folly::Unit> fut = folly::via(&executor_, [this, &hp_list, &finished_count, i, &retire_lists]() { this->Rollout(hp_list, finished_count, i, retire_lists); });
            futures.emplace_back(std::move(fut));
        }
        for (auto itr = futures.begin(); itr != futures.end(); ++itr) {
            auto& fut = *itr;
            std::move(fut).get();
        }
    
        size_t total_rlist_size = 0;
        auto& rl_data = retire_lists.GetLists();
        for (auto itr = rl_data.begin(); itr != rl_data.end(); ++itr) {
            total_rlist_size += itr->second.size();
            hp_list.TryReclaim(itr->second);
        }
        // std::cerr << "Total retire list size: " << total_rlist_size << std::endl;
 
        // find the most visited.
        std::vector<Move> best_actions;
        uint32_t max_visits = 0;
        std::vector<uint64_t>* root_children = (std::vector<uint64_t>*)(root_->children_ & 0x0000ffffffffffffuLL);
        // std::cerr << "Traversing...\n";
        // std::cerr << "Children count: " << root_children->size() << std::endl;
        for (auto itr = root_children->begin(); itr != root_children->end(); ++itr) {
            auto* child = (TreeNode<BOARD_SIZE, WITH_MODEL>*)(*itr & 0x00ffffffffffffff);
            auto idx = (int)(*itr >> 56);
            uint64_t concurrency_visits_score = child->concurrency_visits_score_;
            uint32_t concurrency = (uint32_t)(concurrency_visits_score >> 56);
            uint32_t child_visits = ((uint32_t)(concurrency_visits_score >> 32) & 0xffffffu);
            // std::cerr << "IDX: " << idx << " CONCURRENCY: " << concurrency << " VISITS: " << child_visits << std::endl;
            if (child_visits > max_visits) {
                max_visits = child_visits;
                best_actions.clear();
            }
            if (child_visits == max_visits) {
                best_actions.emplace_back(Move((idx % BOARD_SIZE), idx / BOARD_SIZE));
            }
        }
    
        // std::cerr << "Max visits: " << max_visits << std::endl;
        // std::cerr << "Initializing random device..." << std::endl;
        std::random_device rd;
        // std::cerr << "Initializing engine..." << std::endl;
        std::mt19937 engine(rd());
    
        // std::cerr << "Best action count: " << best_actions.size() << std::endl;
        // std::cerr << "Generating random number..." << std::endl;
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
            static_cast<unsigned int>(task_idx)  // use the task index for extra randomness        };
    */
        thread_local std::mt19937 engine(rd());
        return engine;
    }
    
    template <bool T = WITH_MODEL, typename std::enable_if_t<!T, bool> = true>
    void Rollout(HPList<std::vector<uint64_t>>& hp_list, std::atomic<int>& finished_count, int task_idx, MTRetireLists<std::vector<uint64_t>>& retire_lists) {
        std::vector<std::vector<uint64_t>*> retire_list;
        retire_lists.InheritThreadLocalRetireList(retire_list);
        auto& engine = get_threadlocal_generator();
    
        TreeNode<BOARD_SIZE, WITH_MODEL>* node = root_;
        Move last_move = last_move_;
        bool is_last_black = is_last_black_;
        node->WeakLock();
        while (!node->IsEnd(last_move, is_last_black)) {  // step 3: simulate
            auto elem = node->SelectAndExpand(task_idx, hp_list, retire_list, engine, c_puct_, !is_last_black);  // step 1&2: expand->select
            auto idx = (int)(elem >> 56);
            last_move = { (idx % BOARD_SIZE), (idx / BOARD_SIZE) };
            is_last_black = !is_last_black;
            node = (TreeNode<BOARD_SIZE, WITH_MODEL>*)(elem & 0x00ffffffffffffffuLL);
            node->WeakLock();
        }
        int score = 0;
        if (node->availables_.count() > 0) {
            // score = (is_last_black != is_last_black_) ? 1 : -1;
            score = 1;
        }
        node->UpdateRecursively(score);  // step 4: backpropagate
        int count = finished_count.fetch_add(1) + 1;
        if (__builtin_expect((count % 100000) == 0, 0)) {  // unlikely
            std::cerr << format("Rollout count: %d\n", count); 
        }
        retire_lists.UpdateThreadLocalRetireList(std::move(retire_list));
    }

    template <bool T = WITH_MODEL, typename std::enable_if_t<T, bool> = true>
    void RolloutWithModel(ThreadLocalModels& models, HPList<std::vector<uint64_t>>& hp_list, std::atomic<int>& finished_count, int task_idx, MTRetireLists<std::vector<uint64_t>>& retire_lists) {
        // NOTE(junhaozhang): intra-op OMP parallelism of the forward pass
        // defaults to using all cores; N workers x N OMP threads preempt each
        // other on an N-core box, and measured 1000 sims degrade from ~250ms
        // to ~1300ms (5x+). For a small network, per-forward op
        // synchronization costs far more than the compute, so worker-level
        // parallelism must own the cores exclusively.
        // Both switches are thread-local and affect only this worker, not the
        // batched ops of training on the Python main thread:
        // at::set_num_threads governs ATen's own parallelism;
        // omp_set_num_threads governs the raw OMP region of THNN
        // slow_conv2d's unfolded2d_copy_kernel (gdb shows conv takes this
        // path rather than oneDNN, and that kernel does not read ATen's
        // thread-count setting).
        thread_local bool intraop_threads_set = false;
        if (!intraop_threads_set) {
            at::set_num_threads(1);
            omp_set_num_threads(1);
            intraop_threads_set = true;
        }
        // std::cerr << "RolloutWithModel\n";
        std::vector<std::vector<uint64_t>*> retire_list;
        retire_lists.InheritThreadLocalRetireList(retire_list);
        auto& engine = get_threadlocal_generator();

        TreeNode<BOARD_SIZE, WITH_MODEL>* node = root_;
        Move last_move = last_move_;
        bool is_last_black = is_last_black_;
        node->WeakLock();
        for (; ;) {
            auto* children = (std::vector<uint64_t>*)(node->children_.load() & 0x0000ffffffffffff);
            if (children == nullptr || children->empty()) {
                break;
            }
            // std::cerr << "SelectAndExpand\n";
            auto elem = node->SelectAndExpand(task_idx, hp_list, retire_list, engine, c_puct_, !is_last_black);
            auto idx = (int)(elem >> 56);
            last_move = { (idx % BOARD_SIZE), (idx / BOARD_SIZE) };
            // std::cerr << format("IDX: %d, %d\n", (idx % BOARD_SIZE), (idx / BOARD_SIZE));
            is_last_black = !is_last_black;
            node = (TreeNode<BOARD_SIZE, WITH_MODEL>*)(elem & 0x00ffffffffffffffuLL);
            node->WeakLock();
        }

        auto& model = models.GetThreadLocalModel();
        std::map<int, float> act_probs;
        // NOTE(junhaozhang): this repo's convention is "the score stored in a
        // node is from the perspective of the side that moved INTO the node"
        // (see pure_mcts.hpp scoring +1 at a terminal node, and
        // SelectAndExpand taking max over children).
        // At a terminal node the side that moved into `node` wins, hence
        // score = 1.0.
        float score = 1.0;
        if (!node->IsEnd(last_move, is_last_black)) {
            // std::cerr << "NOT IsEnd!\n";
            torch::Tensor input_tensor = torch::zeros({1, 4, BOARD_SIZE, BOARD_SIZE});
            // std::cerr << "Preparing 1\n";
            node->GenModelInputTensor(input_tensor, last_move, is_last_black);
            // std::cerr << "Preparing 2\n";
            std::vector<torch::jit::IValue> inputs;
            // std::cerr << "Preparing 3\n";
            inputs.push_back(input_tensor);
            // std::cerr << "Infering ...\n";
            auto output_tuple = model.forward(inputs).toTuple();
            // std::cerr << "Getting output 1\n";
            auto policy_output = output_tuple->elements()[0].toTensor().accessor<float, 2>();
            // std::cerr << "Getting output 2\n";
            // The value head outputs the perspective of "the side to move at
            // that state", which is opposite to the perspective of the score
            // stored in this node, so it must be negated.
            score = -output_tuple->elements()[1].toTensor().accessor<float, 2>()[0][0];
            for (int x = 0; x < BOARD_SIZE; ++x) {
                for (int y = 0; y <BOARD_SIZE; ++y) {
                    act_probs[y * BOARD_SIZE + x] = std::exp(policy_output[0][y * BOARD_SIZE + x]);
                }
            }
            // std::cerr << "Expanding...\n";
            node->Expand(act_probs, !is_last_black);
        } else {
            // std::cerr << "IsEnd!\n";
            if (node->availables_.count() == 0) {
                score = 0.0;
            }
        }
        // std::cerr << format("UpdateARecursively:%.2f\n", score);
        node->UpdateRecursively(score);
        int count = finished_count.fetch_add(1) + 1;
        if (__builtin_expect((count % 100000) == 0, 0)) {  // unlikely
            std::cout << format("Rollout count: %d\n", count); 
        }
        // std::cerr << "UpdateThreadLocalRetireList!\n";
        retire_lists.UpdateThreadLocalRetireList(std::move(retire_list));
    }

private:
    folly::CPUThreadPoolExecutor executor_;
    TreeNode<BOARD_SIZE, WITH_MODEL>* root_;
    Move last_move_;
    float c_puct_;
    bool is_last_black_{false};
    bool reuse_tree_states_;
};

}  // namespace gomoku_ai

#endif  // ALPHAZERO_MCTS_HPP_
