#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <utility>
#include <filesystem>
#include <unistd.h>
#include <pybind11/embed.h>  // 需要包含这个头文件


#include "crow.h"
#include "nlohmann/json.hpp"
#include "alphazero_mcts.hpp"

using json = nlohmann::json;
constexpr int BOARD_SIZE = 11;
using PureMCTS = gomoku_ai::GomokuMCTSFramework<BOARD_SIZE, false>;
using AlphaZeroMCTS = gomoku_ai::GomokuMCTSFramework<BOARD_SIZE, true>;
using Move = gomoku_ai::Move;

// 纯 MCTS 无网络先验, 需要大量 rollout 才有棋力; AlphaZero 靠策略价值网络, 1000 次足够。
constexpr int DEFAULT_PURE_MCTS_SIMULATE_TIMES = 1000000;
constexpr int DEFAULT_ALPHAZERO_SIMULATE_TIMES = 1000;
// temperature 越小越接近 argmax(visits), 对局(非自对弈)时取一个很小的值。
constexpr float DEFAULT_ALPHAZERO_TEMPERATURE = 0.001f;
constexpr float DEFAULT_PURE_MCTS_C_PUCT = 2.0f;
constexpr float DEFAULT_ALPHAZERO_C_PUCT = 5.0f;

struct ServerConfig {
    std::string model_path;      // 非空 -> 启用 AlphaZero(MCTS + 策略价值网络); 空 -> 纯 MCTS
    int simulate_times = -1;     // <0 表示按 AI 类型取默认值
    float temperature = DEFAULT_ALPHAZERO_TEMPERATURE;
    float c_puct = -1.0f;        // <0 表示按 AI 类型取默认值
    int cores = 16;
    int port = 7000;
    bool reuse_tree_states = true;

    bool UseModel() const { return !model_path.empty(); }
};

// 在程序启动时初始化Python解释器
bool init_python_environment() {
    if (!Py_IsInitialized()) {
        std::cout << "初始化Python解释器..." << std::endl;
        py::initialize_interpreter();
    } else {
        std::cout << "Python解释器已初始化" << std::endl;
    }

    // 检查GIL状态
    // PyGILState_STATE gstate = PyGILState_GetThisThreadState();
    auto gstate = PyGILState_GetThisThreadState();
    if (gstate == NULL) {
        std::cout << "当前线程没有附加到解释器" << std::endl;
    }

    return Py_IsInitialized();
}

inline bool IsEmptyBoard(const std::vector<std::vector<int>>& board) {
    for (size_t i = 0; i < BOARD_SIZE; ++i) {
        for (size_t j = 0; j < BOARD_SIZE; ++j) {
            if (board[i][j] != 0) {
                return false;
            }
        }
    }
    return true;
}

inline std::mt19937& GetThreadLocalEngine() {
    thread_local std::random_device rd;
    thread_local std::mt19937 engine(rd());
    return engine;
}

// 统一 server 端 AI 的接口: 纯 MCTS 与 AlphaZero(MCTS+策略价值网络) 是两个不同的模板实例,
// 用虚接口擦除类型, 让 GameServer 不用关心具体用哪种 AI。
class AiEngine {
public:
    virtual ~AiEngine() = default;
    virtual bool StateEquals(std::vector<std::vector<int>>& board, bool is_last_black) const = 0;
    virtual void Play(int x, int y) = 0;
    virtual int AvailableCount() const = 0;
    virtual bool IsEnd() const = 0;
    virtual Move SearchBestMove() = 0;
    virtual const char* Kind() const = 0;
    virtual int SimulateTimes() const = 0;
};

class PureMCTSEngine : public AiEngine {
public:
    PureMCTSEngine(const ServerConfig& config, std::vector<std::vector<int>>& board, Move last_move)
        : simulate_times_(config.simulate_times),
          game_(config.cores, board, last_move, config.c_puct, config.reuse_tree_states) {}

    bool StateEquals(std::vector<std::vector<int>>& board, bool is_last_black) const override {
        return game_.StateEquals(board, is_last_black);
    }
    void Play(int x, int y) override { game_.Play(x, y); }
    int AvailableCount() const override { return game_.AvailableCount(); }
    bool IsEnd() const override { return game_.IsEnd(); }
    const char* Kind() const override { return "pure-mcts"; }
    int SimulateTimes() const override { return simulate_times_; }

    Move SearchBestMove() override { return game_.SearchBestMove(simulate_times_); }

private:
    int simulate_times_;
    PureMCTS game_;
};

class AlphaZeroEngine : public AiEngine {
public:
    AlphaZeroEngine(const ServerConfig& config, std::vector<std::vector<int>>& board, Move last_move)
        : simulate_times_(config.simulate_times),
          temperature_(config.temperature),
          model_path_(config.model_path),
          game_(config.cores, board, last_move, config.c_puct, config.reuse_tree_states) {}

    bool StateEquals(std::vector<std::vector<int>>& board, bool is_last_black) const override {
        return game_.StateEquals(board, is_last_black);
    }
    void Play(int x, int y) override { game_.Play(x, y); }
    int AvailableCount() const override { return game_.AvailableCount(); }
    bool IsEnd() const override { return game_.IsEnd(); }
    const char* Kind() const override { return "alphazero"; }
    int SimulateTimes() const override { return simulate_times_; }

    // 与 player.py::AlphaZeroPlayer.get_action 一致: 按 MCTS 访问次数的 softmax 概率采样。
    // temperature 很小时该分布几乎是 one-hot(等价于取访问次数最多的着法)。
    Move SearchBestMove() override {
        auto [sensible_moves, sensible_probs] =
            game_.SearchBestMoveWithModel(simulate_times_, model_path_.c_str(), temperature_);
        if (sensible_moves.empty()) {
            return { -1, -1 };
        }

        size_t argmax = 0;
        double sum = 0.0;
        for (size_t i = 0; i < sensible_probs.size(); ++i) {
            double p = sensible_probs[i];
            if (!std::isfinite(p) || p < 0.0) {
                p = 0.0;
            }
            sum += p;
            if (p > sensible_probs[argmax]) {
                argmax = i;
            }
        }

        size_t picked = argmax;
        if (sum > 0.0) {
            // 概率分布合法才采样, 否则回退到概率最大的着法, 避免 discrete_distribution 未定义行为。
            std::discrete_distribution<size_t> distribution(sensible_probs.begin(), sensible_probs.end());
            picked = distribution(GetThreadLocalEngine());
        } else {
            std::cerr << "WARNING: invalid probs from AlphaZero search, fallback to argmax!" << std::endl;
        }

        int idx = sensible_moves[picked];
        std::cout << "AlphaZero picked prob: " << sensible_probs[picked]
                  << ", max prob: " << sensible_probs[argmax]
                  << ", candidates: " << sensible_moves.size() << std::endl;
        return { idx % BOARD_SIZE, idx / BOARD_SIZE };
    }

private:
    int simulate_times_;
    float temperature_;
    std::string model_path_;
    AlphaZeroMCTS game_;
};

inline std::unique_ptr<AiEngine> CreateEngine(const ServerConfig& config,
                                             std::vector<std::vector<int>>& board,
                                             Move last_move) {
    if (config.UseModel()) {
        return std::make_unique<AlphaZeroEngine>(config, board, last_move);
    }
    return std::make_unique<PureMCTSEngine>(config, board, last_move);
}

class GameServer {
private:
    ServerConfig config_;
    std::unique_ptr<AiEngine> currentGame;
    std::mutex gameMutex;

public:
    explicit GameServer(const ServerConfig& config) : config_(config) {}

    void handleMove(const crow::request& req, crow::json::wvalue& res) {
        std::lock_guard<std::mutex> lock(gameMutex);

        auto data = json::parse(req.body);
        auto boardArr = data["board"].get<std::vector<std::vector<int>>>();
        int x = data["x"];
        int y = data["y"];

        // 处理人类移动
        boardArr[x][y] = 0; // 重置为0，因为前端可能已经设置了值
        std::cout << "Human move: " << x << ", " << y << std::endl;

        // 检查是否需要创建新游戏
        if (!currentGame || !currentGame->StateEquals(boardArr, false)) {
            if (IsEmptyBoard(boardArr)) {
                std::cout << "Initializing a new game!" << std::endl;
            } else {
                std::cout << "WARNING: Re-Initializing the game unexpectedly!" << std::endl;
            }
            boardArr[x][y] = 1;
            currentGame = CreateEngine(config_, boardArr, std::make_pair(x, y));
        } else {
            currentGame->Play(x, y);
        }

        // 检查游戏状态
        if (currentGame->AvailableCount() == 0) {
            std::cout << "Draw!" << std::endl;
            res["result"] = "draw";
            return;
        }

        if (currentGame->IsEnd()) {
            std::cout << "Human win!" << std::endl;
            res["result"] = "win";
            return;
        }

        // AI移动
        int search_times = currentGame->SimulateTimes();
        auto start_time = std::chrono::steady_clock::now();
        auto [ai_x, ai_y] = currentGame->SearchBestMove();
        auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count() / 1000.0;

        std::cout << currentGame->Kind() << " MCTS time cost: " << time_cost
                  << " seconds, search count:" << search_times << std::endl;
        std::cout << "AI move:(" << ai_x << "," << ai_y << ")" << std::endl;

        if (ai_x < 0 || ai_y < 0) {
            std::cout << "ERROR: AI found no available move!" << std::endl;
            res["result"] = "draw";
            return;
        }

        start_time = std::chrono::steady_clock::now();
        currentGame->Play(ai_x, ai_y);
        time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count() / 1000.0;
        std::cout << "Play time cost: " << time_cost << " seconds." << std::endl;

        res["ai_move"] = std::vector<int>{ai_x, ai_y};
        if (currentGame->AvailableCount() == 0) {
            std::cout << "Draw!" << std::endl;
            res["result"] = "draw";
        } else if (currentGame->IsEnd()) {
            std::cout << "AI win!" << std::endl;
            res["result"] = "lose";
        } else {
            res["result"] = "continue";
        }
    }

    void handleRestart(const crow::request& req, crow::json::wvalue& res) {
        std::lock_guard<std::mutex> lock(gameMutex);
        currentGame.reset();
        res["result"] = "ok";
    }

    void serveStaticFiles(crow::SimpleApp& app) {
        // 提供静态文件服务
        CROW_ROUTE(app, "/")
        ([]() {
            crow::mustache::context ctx;
            return crow::mustache::load("index.html").render();
        });

        CROW_ROUTE(app, "/<path>")
        ([](const crow::request& req, crow::response& res, std::string path) {
            res.set_static_file_info("./templates/" + path);
            res.end();
        });
    }
};

void PrintUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [model_path] [options]\n"
        << "\n"
        << "  model_path                TorchScript 模型路径(位置参数, 等价于 --model)。\n"
        << "                            非空 -> 启用 AlphaZero(MCTS + 策略价值网络);\n"
        << "                            为空 -> 使用纯 MCTS。\n"
        << "Options:\n"
        << "  -m, --model <path>        同上\n"
        << "  -n, --simulate-times <n>  MCTS 搜索次数, 默认纯 MCTS " << DEFAULT_PURE_MCTS_SIMULATE_TIMES
        << ", AlphaZero " << DEFAULT_ALPHAZERO_SIMULATE_TIMES << "\n"
        << "  -t, --temperature <f>     AlphaZero 采样温度, 默认 " << DEFAULT_ALPHAZERO_TEMPERATURE << "\n"
        << "      --c-puct <f>          PUCT 常数, 默认纯 MCTS " << DEFAULT_PURE_MCTS_C_PUCT
        << ", AlphaZero " << DEFAULT_ALPHAZERO_C_PUCT << "\n"
        << "  -c, --cores <n>           搜索线程数, 默认 16\n"
        << "  -p, --port <n>            监听端口, 默认 7000\n"
        << "      --no-reuse-tree       不复用上一步的搜索树\n"
        << "  -h, --help                打印本帮助\n"
        << "\n"
        << "NOTE: 模型支持两种形式:\n"
        << "      1) TorchScript(.pt), PolicyValueNet.save_model_with_torchscript() 的产物;\n"
        << "      2) state_dict/checkpoint(.model/.ckpt, 如 current_policy.model),\n"
        << "         启动时自动按内容识别 v1(3conv)/v2(ResNet) 结构并导出成 .pt 再加载。"
        << std::endl;
}

// 支持 "--key value" 与 "--key=value" 两种写法
bool ParseArgs(int argc, char** argv, ServerConfig& config) {
    auto next_value = [&](int& i, const std::string& arg, const char* name, std::string& out) {
        auto eq = arg.find('=');
        if (eq != std::string::npos) {
            out = arg.substr(eq + 1);
            return true;
        }
        if (i + 1 >= argc) {
            std::cerr << "Option " << name << " requires a value!" << std::endl;
            return false;
        }
        out = argv[++i];
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key = arg.substr(0, arg.find('='));
        std::string value;

        if (key == "-h" || key == "--help") {
            PrintUsage(argv[0]);
            return false;
        } else if (key == "-m" || key == "--model") {
            if (!next_value(i, arg, "--model", value)) return false;
            config.model_path = value;
        } else if (key == "-n" || key == "--simulate-times") {
            if (!next_value(i, arg, "--simulate-times", value)) return false;
            config.simulate_times = std::atoi(value.c_str());
        } else if (key == "-t" || key == "--temperature") {
            if (!next_value(i, arg, "--temperature", value)) return false;
            config.temperature = (float)std::atof(value.c_str());
        } else if (key == "--c-puct") {
            if (!next_value(i, arg, "--c-puct", value)) return false;
            config.c_puct = (float)std::atof(value.c_str());
        } else if (key == "-c" || key == "--cores") {
            if (!next_value(i, arg, "--cores", value)) return false;
            config.cores = std::atoi(value.c_str());
        } else if (key == "-p" || key == "--port") {
            if (!next_value(i, arg, "--port", value)) return false;
            config.port = std::atoi(value.c_str());
        } else if (key == "--no-reuse-tree") {
            config.reuse_tree_states = false;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << std::endl;
            PrintUsage(argv[0]);
            return false;
        } else if (config.model_path.empty()) {
            config.model_path = arg;  // 位置参数: 模型路径
        } else {
            std::cerr << "Unexpected argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            return false;
        }
    }

    // 按 AI 类型补默认值
    if (config.simulate_times <= 0) {
        config.simulate_times = config.UseModel() ? DEFAULT_ALPHAZERO_SIMULATE_TIMES
                                                  : DEFAULT_PURE_MCTS_SIMULATE_TIMES;
    }
    if (config.c_puct <= 0.0f) {
        config.c_puct = config.UseModel() ? DEFAULT_ALPHAZERO_C_PUCT : DEFAULT_PURE_MCTS_C_PUCT;
    }
    if (config.cores <= 0) {
        std::cerr << "cores must be positive!" << std::endl;
        return false;
    }
    if (config.UseModel() && config.temperature <= 0.0f) {
        std::cerr << "temperature must be positive!" << std::endl;
        return false;
    }
    return true;
}

// C++ 侧只认 TorchScript(.pt); .model/.ckpt(state_dict/checkpoint)先借助内嵌 Python
// 解释器走 load_net_any_arch 自动识别 v1(3conv)/v2(ResNet, 推断 blocks/channels)结构
// 并导出 .pt。与 elo.py 的 prepare_model_path 同一条路径, 行为保持一致。
// 返回可用的 .pt 路径; 转换失败返回空串。
std::string ExportToTorchScriptIfNeeded(const std::string& model_path) {
    if (model_path.size() >= 3 && model_path.substr(model_path.size() - 3) == ".pt") {
        return model_path;
    }
    std::cout << "Converting state_dict model to TorchScript: " << model_path << std::endl;
    auto ts_path = (std::filesystem::temp_directory_path() /
                    ("web_server_" + std::to_string(::getpid()) + ".pt")).string();
    try {
        py::gil_scoped_acquire gil;
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, ".");  // policy_value_net_pytorch_v2 与 web_server 同目录
        py::module_ pv = py::module_::import("policy_value_net_pytorch_v2");
        py::object net = pv.attr("load_net_any_arch")(BOARD_SIZE, BOARD_SIZE, model_path);
        net.attr("save_model_with_torchscript")(ts_path);
    } catch (const py::error_already_set& e) {
        std::cerr << "Failed to convert model '" << model_path << "': " << e.what() << std::endl;
        return "";
    }
    std::cout << "Exported TorchScript: " << ts_path << std::endl;
    return ts_path;
}

// 提前加载一次模型, 把"模型路径写错/不是 TorchScript 文件"这类问题在启动时就暴露出来,
// 而不是等到第一次落子搜索时才在工作线程里抛异常。
bool CheckModel(const std::string& model_path) {
    std::ifstream fin(model_path, std::ios::binary);
    if (!fin.good()) {
        std::cerr << "Model file not found or unreadable: " << model_path << std::endl;
        return false;
    }
    fin.close();

    try {
        torch::jit::script::Module module = torch::jit::load(model_path);
        auto input = torch::zeros({1, 4, BOARD_SIZE, BOARD_SIZE});
        std::vector<torch::jit::IValue> inputs{input};
        auto output_tuple = module.forward(inputs).toTuple();
        auto policy = output_tuple->elements()[0].toTensor();
        auto value = output_tuple->elements()[1].toTensor();
        if (policy.numel() != BOARD_SIZE * BOARD_SIZE || value.numel() != 1) {
            std::cerr << "Unexpected model output shape: policy numel=" << policy.numel()
                      << ", value numel=" << value.numel() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load TorchScript model '" << model_path << "': " << e.what() << std::endl;
        std::cerr << "HINT: 模型必须由 PolicyValueNet.save_model_with_torchscript() 导出(torch.jit), "
                  << "state_dict(current_policy.model) 无法被 C++ 直接加载。" << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    ServerConfig config;
    if (!ParseArgs(argc, argv, config)) {
        return -1;
    }

    if (!init_python_environment()) {
        std::cerr << "Python interpreter init failed!" << std::endl;
        return -1;
    }

    if (config.UseModel()) {
        config.model_path = ExportToTorchScriptIfNeeded(config.model_path);
        if (config.model_path.empty() || !CheckModel(config.model_path)) {
            return -1;
        }
        std::cout << "AI: AlphaZero(MCTS + policy-value net), model: " << config.model_path
                  << ", simulate times: " << config.simulate_times
                  << ", temperature: " << config.temperature
                  << ", c_puct: " << config.c_puct << std::endl;
    } else {
        std::cout << "AI: pure MCTS(model-free), simulate times: " << config.simulate_times
                  << ", c_puct: " << config.c_puct << std::endl;
    }
    std::cout << "Cores: " << config.cores << ", reuse tree states: "
              << (config.reuse_tree_states ? "true" : "false") << std::endl;

    crow::SimpleApp app;
    GameServer server(config);

    // 设置路由
    CROW_ROUTE(app, "/handle_state")
        .methods("POST"_method)
        ([&server](const crow::request& req) {
            crow::json::wvalue res;
            server.handleMove(req, res);
            return res;
        });

    CROW_ROUTE(app, "/restart")
        .methods("POST"_method)
        ([&server](const crow::request& req) {
            crow::json::wvalue res;
            server.handleRestart(req, res);
            return res;
        });

    // 静态文件服务
    server.serveStaticFiles(app);

    std::cout << "Server running on http://0.0.0.0:" << config.port << std::endl;
    app.port(config.port).multithreaded().run();

    return 0;
}
