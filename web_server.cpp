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
#include <pybind11/embed.h>  // required for the embedded Python interpreter


#include "crow.h"
#include "nlohmann/json.hpp"
#include "alphazero_mcts.hpp"

using json = nlohmann::json;
using Move = gomoku_ai::Move;

// The board size is instantiated at compile time (search tree nodes are
// templates); at runtime --board-size dispatches to the matching instance.
constexpr int DEFAULT_BOARD_SIZE = 11;

// Pure MCTS has no network prior and needs a huge number of rollouts to play
// well; AlphaZero relies on the policy-value net, so 1000 is enough.
constexpr int DEFAULT_PURE_MCTS_SIMULATE_TIMES = 1000000;
constexpr int DEFAULT_ALPHAZERO_SIMULATE_TIMES = 1000;
// The smaller the temperature, the closer to argmax(visits); use a very
// small value for match play (as opposed to self-play).
constexpr float DEFAULT_ALPHAZERO_TEMPERATURE = 0.001f;
constexpr float DEFAULT_PURE_MCTS_C_PUCT = 2.0f;
constexpr float DEFAULT_ALPHAZERO_C_PUCT = 5.0f;

// Number of search worker threads: defaults to the local CPU core count.
inline int DefaultCores() {
    unsigned int n = std::thread::hardware_concurrency();
    return n > 0 ? (int)n : 1;
}

struct ServerConfig {
    std::string model_path;      // non-empty -> AlphaZero (MCTS + policy-value net); empty -> pure MCTS
    int simulate_times = -1;     // <0 means use the default for the AI type
    float temperature = DEFAULT_ALPHAZERO_TEMPERATURE;
    float c_puct = -1.0f;        // <0 means use the default for the AI type
    int cores = DefaultCores();
    int port = 7000;
    int board_size = DEFAULT_BOARD_SIZE;
    bool reuse_tree_states = true;

    bool UseModel() const { return !model_path.empty(); }
};

// Per-game config, changeable via the /new_game API (missing fields keep
// their current values); initial values are inherited from the command line.
// model_path empty -> pure MCTS; non-empty -> AlphaZero (policy-value net).
struct GameConfig {
    std::string model_path;
    int simulate_times = 0;
    float c_puct = 0.0f;
    bool reuse_tree_states = true;
    int cores = DefaultCores();
    int human_color = 1;         // 1 = human plays black and moves first (default), -1 = human plays white (AI moves first)
};

// Initialize the Python interpreter at program start.
bool init_python_environment() {
    if (!Py_IsInitialized()) {
        std::cout << "Initializing Python interpreter..." << std::endl;
        py::initialize_interpreter();
    } else {
        std::cout << "Python interpreter already initialized" << std::endl;
    }

    // Check GIL state
    // PyGILState_STATE gstate = PyGILState_GetThisThreadState();
    auto gstate = PyGILState_GetThisThreadState();
    if (gstate == NULL) {
        std::cout << "Current thread is not attached to the interpreter" << std::endl;
    }

    return Py_IsInitialized();
}

inline bool IsEmptyBoard(const std::vector<std::vector<int>>& board) {
    for (const auto& row : board) {
        for (int v : row) {
            if (v != 0) {
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

// Unified interface for the server-side AI: pure MCTS and AlphaZero
// (MCTS + policy-value net) are two different template instantiations; the
// virtual interface erases the type so GameServer need not care which AI is
// in use.
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

template <int BOARD_SIZE>
class PureMCTSEngine : public AiEngine {
public:
    PureMCTSEngine(const GameConfig& config, std::vector<std::vector<int>>& board, Move last_move)
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
    gomoku_ai::GomokuMCTSFramework<BOARD_SIZE, false> game_;
};

template <int BOARD_SIZE>
class AlphaZeroEngine : public AiEngine {
public:
    AlphaZeroEngine(const GameConfig& config, float temperature, std::vector<std::vector<int>>& board, Move last_move)
        : simulate_times_(config.simulate_times),
          temperature_(temperature),
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

    // Consistent with player.py::AlphaZeroPlayer.get_action: sample from the
    // softmax distribution over MCTS visit counts. With a very small
    // temperature this distribution is almost one-hot (equivalent to picking
    // the most visited move).
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
            // Sample only when the probability distribution is valid;
            // otherwise fall back to the argmax move to avoid undefined
            // behavior of discrete_distribution.
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
    gomoku_ai::GomokuMCTSFramework<BOARD_SIZE, true> game_;
};

template <int BOARD_SIZE>
inline std::unique_ptr<AiEngine> CreateEngine(const GameConfig& config,
                                             float temperature,
                                             std::vector<std::vector<int>>& board,
                                             Move last_move) {
    if (!config.model_path.empty()) {
        return std::make_unique<AlphaZeroEngine<BOARD_SIZE>>(config, temperature, board, last_move);
    }
    return std::make_unique<PureMCTSEngine<BOARD_SIZE>>(config, board, last_move);
}

// Defined near the end of the file (before main); needed by GameServer's
// /new_game.
std::string ExportToTorchScriptIfNeeded(const std::string& model_path, int board_size);
bool CheckModel(const std::string& model_path, int board_size);

// Board-size-agnostic interface exposed to the Crow routes; main
// instantiates GameServerT<11/15> according to --board-size.
class IGameServer {
public:
    virtual ~IGameServer() = default;
    virtual void handleGetConfig(crow::json::wvalue& res) = 0;
    virtual void handleNewGame(const crow::request& req, crow::json::wvalue& res) = 0;
    virtual void handleMove(const crow::request& req, crow::json::wvalue& res) = 0;
    virtual void handleRestart(const crow::request& req, crow::json::wvalue& res) = 0;
};

template <int BOARD_SIZE>
class GameServerT : public IGameServer {
private:
    ServerConfig config_;
    GameConfig gameCfg_;          // current per-game config (changeable by /new_game)
    std::unique_ptr<AiEngine> currentGame;
    std::mutex gameMutex;
    // Single-entry cache for .model/.ckpt -> torchscript conversion: the
    // same source path is never exported twice.
    std::string exportCacheSrc_, exportCacheTs_;

    // Resolve the model path (convert to torchscript and validate if
    // needed); returns an empty string on failure.
    std::string resolveModel(const std::string& path) {
        if (path == exportCacheSrc_) {
            return exportCacheTs_;
        }
        std::string ts = ExportToTorchScriptIfNeeded(path, BOARD_SIZE);
        if (ts.empty() || !CheckModel(ts, BOARD_SIZE)) {
            return "";
        }
        exportCacheSrc_ = path;
        exportCacheTs_ = ts;
        return ts;
    }

    // Opening when the AI (black) moves first: build the engine on an empty
    // board, search and play the first move.
    void aiOpenMove(crow::json::wvalue& res) {
        std::vector<std::vector<int>> empty(BOARD_SIZE, std::vector<int>(BOARD_SIZE, 0));
        currentGame = CreateEngine<BOARD_SIZE>(gameCfg_, config_.temperature, empty, { -1, -1 });
        auto [ax, ay] = currentGame->SearchBestMove();
        currentGame->Play(ax, ay);
        std::cout << "AI opens(black): (" << ax << "," << ay << ")" << std::endl;
        res["ai_move"] = std::vector<int>{ax, ay};
    }

public:
    explicit GameServerT(const ServerConfig& config) : config_(config) {
        gameCfg_.model_path = config.model_path;
        gameCfg_.simulate_times = config.simulate_times;
        gameCfg_.c_puct = config.c_puct;
        gameCfg_.reuse_tree_states = config.reuse_tree_states;
        gameCfg_.cores = config.cores;
        gameCfg_.human_color = 1;
    }

    // The frontend fetches the default config to prefill the form.
    void handleGetConfig(crow::json::wvalue& res) override {
        std::lock_guard<std::mutex> lock(gameMutex);
        res["board_size"] = BOARD_SIZE;
        res["model"] = config_.model_path;
        res["simulate_times"] = gameCfg_.simulate_times;
        res["c_puct"] = gameCfg_.c_puct;
        res["reuse_states"] = gameCfg_.reuse_tree_states;
        res["cores"] = gameCfg_.cores;
        res["human_color"] = gameCfg_.human_color;
        res["ai_type"] = gameCfg_.model_path.empty() ? "pure" : "model";
    }

    // Start a new game: optionally carries a new config
    // (ai_type/model/simulate_times/c_puct/reuse_states/cores/human_color);
    // missing fields keep their current values. When the human plays white,
    // the AI moves first and ai_move is returned.
    void handleNewGame(const crow::request& req, crow::json::wvalue& res) override {
        std::lock_guard<std::mutex> lock(gameMutex);
        if (!req.body.empty()) {
            auto data = json::parse(req.body, nullptr, false);
            if (data.is_discarded()) {
                res["result"] = "error";
                res["message"] = "invalid json body";
                return;
            }
            if (data.contains("ai_type")) {
                std::string t = data["ai_type"].get<std::string>();
                if (t == "pure") {
                    gameCfg_.model_path.clear();
                } else if (t == "model" && gameCfg_.model_path.empty()
                           && !(data.contains("model") && !data["model"].get<std::string>().empty())) {
                    res["result"] = "error";
                    res["message"] = "ai_type=model but no model path was provided (nor given on the command line)";
                    return;
                }
            }
            if (data.contains("model")) {
                std::string m = data["model"].get<std::string>();
                if (!m.empty()) {
                    std::string ts = resolveModel(m);
                    if (ts.empty()) {
                        res["result"] = "error";
                        res["message"] = "failed to load model: " + m;
                        return;
                    }
                    gameCfg_.model_path = ts;
                }
            }
            if (data.contains("simulate_times")) {
                gameCfg_.simulate_times = data["simulate_times"].get<int>();
            }
            if (data.contains("c_puct")) {
                gameCfg_.c_puct = (float)data["c_puct"].get<double>();
            }
            if (data.contains("reuse_states")) {
                gameCfg_.reuse_tree_states = data["reuse_states"].get<bool>();
            }
            if (data.contains("cores")) {
                gameCfg_.cores = data["cores"].get<int>();
            }
            if (data.contains("human_color")) {
                int c = data["human_color"].get<int>();
                if (c != 1 && c != -1) {
                    res["result"] = "error";
                    res["message"] = "human_color must be 1 (black) or -1 (white)";
                    return;
                }
                gameCfg_.human_color = c;
            }
        }
        currentGame.reset();
        std::cout << "New game: ai=" << (gameCfg_.model_path.empty() ? "pure-mcts" : gameCfg_.model_path)
                  << ", sims=" << gameCfg_.simulate_times << ", c_puct=" << gameCfg_.c_puct
                  << ", cores=" << gameCfg_.cores << ", reuse=" << gameCfg_.reuse_tree_states
                  << ", human_color=" << gameCfg_.human_color << std::endl;
        res["result"] = "ok";
        if (gameCfg_.human_color == -1) {
            aiOpenMove(res);  // human plays white; AI plays black and moves first
        }
    }

    void handleMove(const crow::request& req, crow::json::wvalue& res) override {
        std::lock_guard<std::mutex> lock(gameMutex);

        auto data = json::parse(req.body);
        auto boardArr = data["board"].get<std::vector<std::vector<int>>>();
        int x = data["x"];
        int y = data["y"];

        // Handle the human move
        boardArr[x][y] = 0; // reset to 0 because the frontend may have already set it
        std::cout << "Human move: " << x << ", " << y << std::endl;

        // Check whether a new game needs to be created.
        // At this point the last move on the board was made by the AI, whose
        // color = -human_color, so:
        // human plays black (human_color=1) -> the last move was white;
        // human plays white -> the last move was black.
        bool last_black = (gameCfg_.human_color == -1);
        if (!currentGame || !currentGame->StateEquals(boardArr, last_black)) {
            if (IsEmptyBoard(boardArr)) {
                std::cout << "Initializing a new game!" << std::endl;
            } else {
                std::cout << "WARNING: Re-Initializing the game unexpectedly!" << std::endl;
            }
            boardArr[x][y] = gameCfg_.human_color;  // apply the human's move (black 1 / white -1)
            currentGame = CreateEngine<BOARD_SIZE>(gameCfg_, config_.temperature, boardArr, std::make_pair(x, y));
        } else {
            currentGame->Play(x, y);
        }

        // Check the game state
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

        // AI move
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

    // Restart a game with the current config; when the human plays white,
    // the AI moves first and ai_move is returned.
    void handleRestart(const crow::request& req, crow::json::wvalue& res) override {
        std::lock_guard<std::mutex> lock(gameMutex);
        currentGame.reset();
        res["result"] = "ok";
        if (gameCfg_.human_color == -1) {
            aiOpenMove(res);
        }
    }

};

void ServeStaticFiles(crow::SimpleApp& app) {
    // Serve static files
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

void PrintUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [model_path] [options]\n"
        << "\n"
        << "  model_path                Path to a TorchScript model (positional, equivalent to --model).\n"
        << "                            Non-empty -> AlphaZero (MCTS + policy-value net);\n"
        << "                            empty -> pure MCTS.\n"
        << "Options:\n"
        << "  -m, --model <path>        Same as above\n"
        << "  -n, --simulate-times <n>  MCTS simulations per move; defaults: pure MCTS "
        << DEFAULT_PURE_MCTS_SIMULATE_TIMES
        << ", AlphaZero " << DEFAULT_ALPHAZERO_SIMULATE_TIMES << "\n"
        << "  -t, --temperature <f>     AlphaZero sampling temperature, default "
        << DEFAULT_ALPHAZERO_TEMPERATURE << "\n"
        << "      --c-puct <f>          PUCT constant; defaults: pure MCTS " << DEFAULT_PURE_MCTS_C_PUCT
        << ", AlphaZero " << DEFAULT_ALPHAZERO_C_PUCT << "\n"
        << "  -c, --cores <n>           Number of search threads, default: local CPU count\n"
        << "  -p, --port <n>            Listening port, default 7000\n"
        << "  -s, --board-size <n>      Board edge length, 11 or 15, default " << DEFAULT_BOARD_SIZE << "\n"
        << "      --no-reuse-tree       Do not reuse the search tree from the previous move\n"
        << "  -h, --help                Print this help\n"
        << "\n"
        << "NOTE: two model forms are supported:\n"
        << "      1) TorchScript (.pt), produced by PolicyValueNet.save_model_with_torchscript();\n"
        << "      2) state_dict/checkpoint (.model/.ckpt, e.g. current_policy.model) --\n"
        << "         at startup the v1 (3conv) / v2 (ResNet) architecture is auto-detected\n"
        << "         from the content and the weights are exported to .pt before loading."
        << std::endl;
}

// Supports both "--key value" and "--key=value" styles.
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
        } else if (key == "-s" || key == "--board-size") {
            if (!next_value(i, arg, "--board-size", value)) return false;
            config.board_size = std::atoi(value.c_str());
        } else if (key == "--no-reuse-tree") {
            config.reuse_tree_states = false;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << std::endl;
            PrintUsage(argv[0]);
            return false;
        } else if (config.model_path.empty()) {
            config.model_path = arg;  // positional argument: model path
        } else {
            std::cerr << "Unexpected argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            return false;
        }
    }

    // Fill in defaults according to the AI type
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
    if (config.board_size != 11 && config.board_size != 15) {
        std::cerr << "board_size must be 11 or 15 (got " << config.board_size << ")!" << std::endl;
        return false;
    }
    if (config.UseModel() && config.temperature <= 0.0f) {
        std::cerr << "temperature must be positive!" << std::endl;
        return false;
    }
    return true;
}

// The C++ side only accepts TorchScript (.pt); .model/.ckpt
// (state_dict/checkpoint) first go through the embedded Python interpreter
// via load_net_any_arch, which auto-detects the v1 (3conv) / v2 (ResNet,
// inferring blocks/channels) architecture and exports a .pt. This is the
// same code path as elo.py's prepare_model_path, so behavior stays
// consistent.
// Returns a usable .pt path; an empty string on conversion failure.
std::string ExportToTorchScriptIfNeeded(const std::string& model_path, int board_size) {
    if (model_path.size() >= 3 && model_path.substr(model_path.size() - 3) == ".pt") {
        return model_path;
    }
    std::cout << "Converting state_dict model to TorchScript: " << model_path << std::endl;
    auto ts_path = (std::filesystem::temp_directory_path() /
                    ("web_server_" + std::to_string(::getpid()) + ".pt")).string();
    try {
        py::gil_scoped_acquire gil;
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, ".");  // policy_value_net_pytorch_v2 lives next to web_server
        py::module_ pv = py::module_::import("policy_value_net_pytorch_v2");
        py::object net = pv.attr("load_net_any_arch")(board_size, board_size, model_path);
        net.attr("save_model_with_torchscript")(ts_path);
    } catch (const py::error_already_set& e) {
        std::cerr << "Failed to convert model '" << model_path << "': " << e.what() << std::endl;
        return "";
    }
    std::cout << "Exported TorchScript: " << ts_path << std::endl;
    return ts_path;
}

// Load the model once up front, so that problems like "wrong model path /
// not a TorchScript file" surface at startup instead of being raised in a
// worker thread at the first move search.
bool CheckModel(const std::string& model_path, int board_size) {
    std::ifstream fin(model_path, std::ios::binary);
    if (!fin.good()) {
        std::cerr << "Model file not found or unreadable: " << model_path << std::endl;
        return false;
    }
    fin.close();

    try {
        torch::jit::script::Module module = torch::jit::load(model_path);
        auto input = torch::zeros({1, 4, board_size, board_size});
        std::vector<torch::jit::IValue> inputs{input};
        auto output_tuple = module.forward(inputs).toTuple();
        auto policy = output_tuple->elements()[0].toTensor();
        auto value = output_tuple->elements()[1].toTensor();
        if (policy.numel() != board_size * board_size || value.numel() != 1) {
            std::cerr << "Unexpected model output shape: policy numel=" << policy.numel()
                      << ", value numel=" << value.numel() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load TorchScript model '" << model_path << "': " << e.what() << std::endl;
        std::cerr << "HINT: the model must be exported by PolicyValueNet.save_model_with_torchscript() (torch.jit); "
                  << "a state_dict (current_policy.model) cannot be loaded by C++ directly." << std::endl;
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
        config.model_path = ExportToTorchScriptIfNeeded(config.model_path, config.board_size);
        if (config.model_path.empty() || !CheckModel(config.model_path, config.board_size)) {
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
    std::cout << "Board size: " << config.board_size << "x" << config.board_size
              << ", cores: " << config.cores << ", reuse tree states: "
              << (config.reuse_tree_states ? "true" : "false") << std::endl;

    crow::SimpleApp app;
    // The board size is a compile-time parameter of the search-tree
    // template; dispatch on the runtime config here.
    std::unique_ptr<IGameServer> server_ptr;
    if (config.board_size == 11) {
        server_ptr = std::make_unique<GameServerT<11>>(config);
    } else {
        server_ptr = std::make_unique<GameServerT<15>>(config);
    }
    IGameServer& server = *server_ptr;

    // Set up routes
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

    CROW_ROUTE(app, "/new_game")
        .methods("POST"_method)
        ([&server](const crow::request& req) {
            crow::json::wvalue res;
            server.handleNewGame(req, res);
            return res;
        });

    CROW_ROUTE(app, "/config")
        ([&server](const crow::request& req) {
            crow::json::wvalue res;
            server.handleGetConfig(res);
            return res;
        });

    // Static file service
    ServeStaticFiles(app);

    // NOTE(junhaozhang): the main thread's GIL must be released before
    // entering the Crow event loop! After py::initialize_interpreter() the
    // main thread keeps holding the GIL; without releasing it, the
    // py::gil_scoped_acquire in /new_game (model conversion) on a Crow
    // worker thread would block forever (observed deadlock in practice).
    py::gil_scoped_release gil_release;

    std::cout << "Server running on http://0.0.0.0:" << config.port << std::endl;
    app.port(config.port).multithreaded().run();

    return 0;
}
