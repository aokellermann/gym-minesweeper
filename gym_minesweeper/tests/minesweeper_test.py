"""Dummy."""

import numpy.testing as npt

from gym_minesweeper import MinesweeperEnv, SPACE_UNKNOWN


def test_no_mines_init():
    size = (2, 3)
    ms = MinesweeperEnv(size, 0)
    assert size == ms.board_size
    assert 0 == ms.num_mines
    npt.assert_array_equal([], ms.hist)
    npt.assert_array_equal([[SPACE_UNKNOWN] * size[1]] * size[0], ms.board)


def test_no_mines_step():
    size = (2, 3)
    ms = MinesweeperEnv(size, 0)
    action = (0, 0)
    board, reward, done, info = ms.step(action)

    expected_board = [[0] * size[1]] * size[0]
    npt.assert_array_equal(ms.board, expected_board)
    npt.assert_array_equal(ms.hist, [action])

    npt.assert_array_equal(board, expected_board)
    assert reward == 1000
    assert done
    assert info == dict()


def test_mines_step():
    size = (4, 5)
    ms = MinesweeperEnv(size, 3)
    ms.seed(42069)
    action = (0, 0)
    board, reward, done, info = ms.step(action)

    expected_board = [[0, 1, -1, -1, -1], [0, 1, -1, -1, -1], [1, 1, -1, -1, -1], [-1, -1, -1, -1, -1]]
    npt.assert_array_equal(ms.board, expected_board)
    npt.assert_array_equal(ms.hist, [action])

    npt.assert_array_equal(board, expected_board)
    assert reward == 5
    assert not done
    assert not info


def assert_game(ms, actions, expected_boards, expected_rewards, expected_dones, expected_infos):
    expected_hist = []
    for i in range(len(actions)):
        board, reward, done, info = ms.step(actions[i])

        def err_msg():
            return "idx: {}".format(i)

        npt.assert_array_equal(ms.board, expected_boards[i], err_msg())
        npt.assert_array_equal(board, expected_boards[i], err_msg())

        expected_hist.append(actions[i])
        npt.assert_array_equal(ms.hist, expected_hist)

        assert reward == expected_rewards[i], err_msg()
        assert done == expected_dones[i], err_msg()
        assert info == expected_infos[i], err_msg()


def test_win():
    size = (4, 5)
    ms = MinesweeperEnv(size, 3)
    ms.seed(42069)

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
    expected_infos = [dict()] * len(expected_boards)

    assert_game(ms, actions, expected_boards, expected_rewards, expected_dones, expected_infos)
