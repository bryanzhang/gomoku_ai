#include "alphazero_mcts.hpp"
#include <pybind11/functional.h>

PYBIND11_MODULE(gomoku_ai, m) {
    m.doc() = "Pybind11 pure MCTS gomoku plugin";

    // pure mcts (model-free)
    py::class_<gomoku_ai::GomokuMCTSFramework<8, false>>(m, "PureMCTSFramework8")
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
        .def("StateEquals", &gomoku_ai::GomokuMCTSFramework<8, false>::NpStateEquals, py::arg("np_board"), py::arg("s_last_black"), "Is the state equals the game's state\n")
        .def("Play", &gomoku_ai::GomokuMCTSFramework<8, false>::Play, py::arg("x"), py::arg("y"),
            "Make a move.\n")
        .def("AvailableCount", &gomoku_ai::GomokuMCTSFramework<8, false>::AvailableCount, "How many grids there are to put pieces\n")
        .def("IsEnd", &gomoku_ai::GomokuMCTSFramework<8, false>::IsEnd, "Is the game over(win,lose or draw)\n")
        .def("SearchBestMove", &gomoku_ai::GomokuMCTSFramework<8, false>::SearchBestMove<>, py::arg("simulate_times"), "Monto-Carlo tree search for specified times to search the best move.\n");

    py::class_<gomoku_ai::GomokuMCTSFramework<9, false>>(m, "PureMCTSFramework9")
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
        .def("StateEquals", &gomoku_ai::GomokuMCTSFramework<9, false>::NpStateEquals, py::arg("board"), py::arg("s_last_black"), "Is the state equals the game's state")
        .def("Play", &gomoku_ai::GomokuMCTSFramework<9, false>::Play, py::arg("x"), py::arg("y"),
            "Make a move.\n")
        .def("AvailableCount", &gomoku_ai::GomokuMCTSFramework<9, false>::AvailableCount, "How many grids there are to put pieces\n")
        .def("IsEnd", &gomoku_ai::GomokuMCTSFramework<9, false>::IsEnd, "Is the game over(win,lose or draw)\n")
        .def("SearchBestMove", &gomoku_ai::GomokuMCTSFramework<9, false>::SearchBestMove<>, py::arg("simulate_times"), "Monto-Carlo tree search for specified times to search the best move.\n");

    py::class_<gomoku_ai::GomokuMCTSFramework<11, false>>(m, "PureMCTSFramework11")
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
        .def("StateEquals", &gomoku_ai::GomokuMCTSFramework<11, false>::NpStateEquals, py::arg("board"), py::arg("s_last_black"), "Is the state equals the game's state")
        .def("Play", &gomoku_ai::GomokuMCTSFramework<11, false>::Play, py::arg("x"), py::arg("y"),
            "Make a move.\n")
        .def("AvailableCount", &gomoku_ai::GomokuMCTSFramework<11, false>::AvailableCount, "How many grids there are to put pieces\n")
        .def("IsEnd", &gomoku_ai::GomokuMCTSFramework<11, false>::IsEnd, "Is the game over(win,lose or draw)\n")
        .def("SearchBestMove", &gomoku_ai::GomokuMCTSFramework<11, false>::SearchBestMove<>, py::arg("simulate_times"), "Monto-Carlo tree search for specified times to search the best move.\n");

    py::class_<gomoku_ai::GomokuMCTSFramework<15, false>>(m, "PureMCTSFramework15")
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
        .def("StateEquals", &gomoku_ai::GomokuMCTSFramework<15, false>::NpStateEquals, py::arg("board"), py::arg("s_last_black"), "Is the state equals the game's state")
        .def("Play", &gomoku_ai::GomokuMCTSFramework<15, false>::Play, py::arg("x"), py::arg("y"),
            "Make a move.\n")
        .def("AvailableCount", &gomoku_ai::GomokuMCTSFramework<15, false>::AvailableCount, "How many grids there are to put pieces\n")
        .def("IsEnd", &gomoku_ai::GomokuMCTSFramework<15, false>::IsEnd, "Is the game over(win,lose or draw)\n")
        .def("SearchBestMove", &gomoku_ai::GomokuMCTSFramework<15, false>::SearchBestMove<>, py::arg("simulate_times"), "Monto-Carlo tree search for specified times to search the best move.\n");

    // MCTS with model
    py::class_<gomoku_ai::GomokuMCTSFramework<8, true>>(m, "AlphaZeroMCTSFramework8")
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
        .def("StateEquals", &gomoku_ai::GomokuMCTSFramework<8, true>::NpStateEquals, py::arg("np_board"), py::arg("s_last_black"), "Is the state equals the game's state\n")
        .def("Play", &gomoku_ai::GomokuMCTSFramework<8, true>::Play, py::arg("x"), py::arg("y"),
            "Make a move.\n")
        .def("AvailableCount", &gomoku_ai::GomokuMCTSFramework<8, true>::AvailableCount, "How many grids there are to put pieces\n")
        .def("IsEnd", &gomoku_ai::GomokuMCTSFramework<8, true>::IsEnd, "Is the game over(win,lose or draw)\n")
        .def("SearchBestMove", &gomoku_ai::GomokuMCTSFramework<8, true>::SearchBestMoveWithModel<>, py::arg("simulate_times"), py::arg("model_path"), py::arg("temperature"), "Monto-Carlo tree search for specified times to search the best move.\n");

    py::class_<gomoku_ai::GomokuMCTSFramework<11, true>>(m, "AlphaZeroMCTSFramework11")
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
        .def("StateEquals", &gomoku_ai::GomokuMCTSFramework<11, true>::NpStateEquals, py::arg("np_board"), py::arg("s_last_black"), "Is the state equals the game's state\n")
        .def("Play", &gomoku_ai::GomokuMCTSFramework<11, true>::Play, py::arg("x"), py::arg("y"),
            "Make a move.\n")
        .def("AvailableCount", &gomoku_ai::GomokuMCTSFramework<11, true>::AvailableCount, "How many grids there are to put pieces\n")
        .def("IsEnd", &gomoku_ai::GomokuMCTSFramework<11, true>::IsEnd, "Is the game over(win,lose or draw)\n")
        .def("SearchBestMove", &gomoku_ai::GomokuMCTSFramework<11, true>::SearchBestMoveWithModel<>, py::arg("simulate_times"), py::arg("model_path"), py::arg("temperature"), "Monto-Carlo tree search for specified times to search the best move.\n");
}
