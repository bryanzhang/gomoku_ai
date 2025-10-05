#include "gomoku_ai.hpp"

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
