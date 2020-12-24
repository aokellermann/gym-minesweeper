"""Dummy."""

import numpy.testing as npt

from gym_minesweeper import MinesweeperEnv, SPACE_UNKNOWN


def test_no_mines_init():
    """Asserts that initializing with no mines works properly"""

    size = (30, 50)
    ms_game = MinesweeperEnv(size, 0)
    assert size == ms_game.board_size
    assert ms_game.num_mines == 0
    npt.assert_array_equal([], ms_game.hist)
    npt.assert_array_equal([[SPACE_UNKNOWN] * size[1]] * size[0], ms_game.board)


def test_no_mines_step():
    """Asserts that taking one step with no mines wins"""

    size = (30, 50)
    ms_game = MinesweeperEnv(size, 0)
    action = (21, 5)
    board, reward, done, info = ms_game.step(action)

    expected_board = [[0] * size[1]] * size[0]
    npt.assert_array_equal(ms_game.board, expected_board)
    npt.assert_array_equal(ms_game.hist, [action])

    npt.assert_array_equal(board, expected_board)
    assert reward == 1000
    assert done
    assert info == dict()


def assert_game(ms_game, actions, expected_boards, expected_rewards, expected_dones):
    """Given a full list of game steps, plays through the game and asserts all states are correct."""

    expected_hist = []

    def err_msg(idx):
        return "idx: {}".format(idx)

    for i, action in enumerate(actions):
        board, reward, done, info = ms_game.step(action)

        npt.assert_array_equal(ms_game.board, expected_boards[i], err_msg(i))
        npt.assert_array_equal(board, expected_boards[i], err_msg(i))

        expected_hist.append(action)
        npt.assert_array_equal(ms_game.hist, expected_hist, err_msg(i))

        assert reward == expected_rewards[i], err_msg(i)
        assert done == expected_dones[i], err_msg(i)
        assert info == dict(), err_msg(i)


def test_win():
    """Asserts that a winning playthrough works."""

    size = (4, 5)
    ms_game = MinesweeperEnv(size, 3)
    ms_game.seed(42069)

    actions = [(0, 0), (3, 3), (0, 3), (1, 2), (0, 4), (1, 4)]
    expected_boards = [
        [[0, 1, -1, -1, -1], [0, 1, -1, -1, -1], [1, 1, -1, -1, -1], [-1, -1, -1, -1, -1]],
        [[0, 1, -1, -1, -1], [0, 1, -1, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, -1], [0, 1, -1, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, -1], [0, 1, 2, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, 1], [0, 1, 2, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, 1], [0, 1, 2, -1, 1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
    ]

    expected_rewards = [5] * (len(expected_boards) - 1) + [1000]
    expected_dones = [False] * (len(expected_boards) - 1) + [True]

    assert_game(ms_game, actions, expected_boards, expected_rewards, expected_dones)
