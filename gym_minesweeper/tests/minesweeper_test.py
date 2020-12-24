"""Dummy."""

from gym_minesweeper import MinesweeperEnv, SPACE_MINE, SPACE_UNKNOWN

import numpy as np
import numpy.testing as npt


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
    npt.assert_array_equal(expected_board, ms.board)
    npt.assert_array_equal([action], ms.hist)

    npt.assert_array_equal(expected_board, board)
    assert 1000 == reward
    assert done
    npt.assert_array_equal(expected_board, info["master"])
