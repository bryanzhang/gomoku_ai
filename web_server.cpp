#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

#include "crow.h"
#include "nlohmann/json.hpp"
#include "gomoku_ai.hpp"

using json = nlohmann::json;
constexpr int BOARD_SIZE = 11;
using GomokuAI = gomoku_ai::PureMCTSGame<BOARD_SIZE>;

inline bool IsEmptyBoard(const py::array_t<int32_t>& board) {
    py::buffer_info buf_info = board.request();
    int32_t* data = static_cast<int32_t*>(buf_info.ptr);
    size_t total_size = buf_info.size;
    for (size_t i = 0; i < total_size; ++i) {
        if (data[i] != 0) {
            return false;
        }
    }
    return true;
}

class GameServer {
private:
    std::unique_ptr<GomokuAI> currentGame;
    std::mutex gameMutex;
    static const int CORES = 1;
    
public:
    void handleMove(const crow::request& req, crow::response& res) {
        std::lock_guard<std::mutex> lock(gameMutex);
        
        try {
            auto data = json::parse(req.body);
            auto originalBoardArr = data["board"].get<std::vector<std::vector<int>>>();
            int x = data["x"];
            int y = data["y"];
            auto boardArr = py::array_t<int32_t>({BOARD_SIZE, BOARD_SIZE});

            // 处理人类移动
            originalBoardArr[x][y] = 0; // 重置为0，因为前端可能已经设置了值
            int32_t* ptr = static_cast<int32_t*>(boardArr.request().ptr);
            for (int i = 0; i < BOARD_SIZE; ++i) {
                for (int j = 0; j < BOARD_SIZE; ++j) {
                    ptr[i * BOARD_SIZE + j] = originalBoardArr[i][j];
                }
            }
            std::cout << "Human move: " << x << ", " << y << std::endl;
            
            
            // 检查是否需要创建新游戏
            if (!currentGame || !currentGame->StateEquals(boardArr, false)) {
                if (IsEmptyBoard(boardArr)) {
                    std::cout << "Initializing a new game!" << std::endl;
                } else {
                    std::cout << "WARNING: Re-Initializing the game unexpectedly!" << std::endl;
                }
                ptr[x * BOARD_SIZE + y] = 1;
                currentGame = std::make_unique<GomokuAI>(CORES, boardArr, 
                                                         std::make_pair(x, y), 2.0, true);
            } else {
                currentGame->Play(x, y);
            }
            
            // 检查游戏状态
            if (currentGame->AvailableCount() == 0) {
                std::cout << "Draw!" << std::endl;
                json response = {{"result", "draw"}};
                res.write(response.dump());
                res.end();
                return;
            }
            
            if (currentGame->IsEnd()) {
                std::cout << "Human win!" << std::endl;
                json response = {{"result", "win"}};
                res.write(response.dump());
                res.end();
                return;
            }
            
            // AI移动
            int search_times = 100000;
            auto start_time = std::chrono::steady_clock::now();
            auto [ai_x, ai_y] = currentGame->SearchBestMove(search_times);
            auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count() / 1000.0;
                
            std::cout << "MCTS time cost: " << time_cost << " seconds, search count:" 
                      << search_times << std::endl;
            std::cout << "AI move:(" << ai_x << "," << ai_y << ")" << std::endl;
            
            currentGame->Play(ai_x, ai_y);
            
            json response;
            if (currentGame->AvailableCount() == 0) {
                std::cout << "Draw!" << std::endl;
                response = {{"result", "draw"}, {"ai_move", {ai_x, ai_y}}};
            } else if (currentGame->IsEnd()) {
                std::cout << "AI win!" << std::endl;
                response = {{"result", "lose"}, {"ai_move", {ai_x, ai_y}}};
            } else {
                response = {{"result", "continue"}, {"ai_move", {ai_x, ai_y}}};
            }
            
            res.write(response.dump());
            
        } catch (const std::exception& e) {
            json error = {{"error", e.what()}};
            res.code = 400;
            res.write(error.dump());
        }
        
        res.end();
    }
    
    void handleRestart(const crow::request& req, crow::response& res) {
        std::lock_guard<std::mutex> lock(gameMutex);
        currentGame.reset();
        json response = {{"result", "ok"}};
        res.write(response.dump());
        res.end();
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
    
private:
    bool isEmptyBoard(const std::vector<std::vector<int>>& board) {
        for (const auto& row : board) {
            for (int cell : row) {
                if (cell != 0) return false;
            }
        }
        return true;
    }
};

int main() {
    crow::SimpleApp app;
    GameServer server;
    
    // 设置路由
    CROW_ROUTE(app, "/handle_state")
        .methods("POST"_method)
        ([&server](const crow::request& req) {
            crow::response res;
            server.handleMove(req, res);
            return res;
        });
    
    CROW_ROUTE(app, "/restart")
        .methods("POST"_method)
        ([&server](const crow::request& req) {
            crow::response res;
            server.handleRestart(req, res);
            return res;
        });
    
    // 静态文件服务
    server.serveStaticFiles(app);
    
    std::cout << "Server running on http://0.0.0.0:7000" << std::endl;
    app.port(7000).multithreaded().run();
    
    return 0;
}
