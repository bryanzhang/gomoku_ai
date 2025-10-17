#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <pybind11/embed.h>  // 需要包含这个头文件


#include "crow.h"
#include "nlohmann/json.hpp"
#include "alphazero_mcts.hpp"

using json = nlohmann::json;
constexpr int BOARD_SIZE = 11;
using GomokuAI = gomoku_ai::GomokuMCTSFramework<BOARD_SIZE, false>;

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

class GameServer {
private:
    std::unique_ptr<GomokuAI> currentGame;
    std::mutex gameMutex;
    static const int CORES = 20;
    
public:
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
            currentGame = std::make_unique<GomokuAI>(CORES, boardArr, 
                                                     std::make_pair(x, y), 2.0, true);
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
        int search_times = 1500000;
        auto start_time = std::chrono::steady_clock::now();
        auto [ai_x, ai_y] = currentGame->SearchBestMove(search_times);
        auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count() / 1000.0;
            
        std::cout << "MCTS time cost: " << time_cost << " seconds, search count:" 
                  << search_times << std::endl;
        std::cout << "AI move:(" << ai_x << "," << ai_y << ")" << std::endl;
        
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

int main() {
    if (!init_python_environment()) {
        std::cerr << "Python interpreter init failed!" << std::endl;
        return -1;
    }

    crow::SimpleApp app;
    GameServer server;
    
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
    
    std::cout << "Server running on http://0.0.0.0:7000" << std::endl;
    app.port(7000).multithreaded().run();
    
    return 0;
}
