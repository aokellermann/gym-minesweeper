"""Tests for minesweeper env implementation."""
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
from PIL import Image

from gym_minesweeper import MinesweeperEnv, SPACE_UNKNOWN, REWARD_WIN, REWARD_LOSE, REWARD_CLEAR

TEST_BOARD_SIZE = (4, 5)
TEST_NUM_MINES = 3
TEST_SEED = 42069


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
    assert reward == REWARD_WIN
    assert done
    assert info == dict()


def create_game():
    """Creates a deterministic 4x5 game"""
    size = TEST_BOARD_SIZE
    ms_game = MinesweeperEnv(size, TEST_NUM_MINES)
    ms_game.seed(TEST_SEED)
    return ms_game


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


def test_win(ms_game=create_game()):
    """Asserts that a winning playthrough works."""

    actions = [(0, 0), (3, 3), (0, 3), (1, 2), (0, 4), (1, 4)]
    expected_boards = [
        [[0, 1, -1, -1, -1], [0, 1, -1, -1, -1], [1, 1, -1, -1, -1], [-1, -1, -1, -1, -1]],
        [[0, 1, -1, -1, -1], [0, 1, -1, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, -1], [0, 1, -1, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, -1], [0, 1, 2, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, 1], [0, 1, 2, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, 1], [0, 1, 2, -1, 1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
    ]

    expected_rewards = [REWARD_CLEAR] * (len(expected_boards) - 1) + [REWARD_WIN]
    expected_dones = [False] * (len(expected_boards) - 1) + [True]

    assert_game(ms_game, actions, expected_boards, expected_rewards, expected_dones)


def test_lose(ms_game=create_game()):
    """Asserts that a losing playthrough works."""

    actions = [(0, 0), (3, 3), (0, 3), (1, 2), (0, 4), (0, 2)]
    expected_boards = [
        [[0, 1, -1, -1, -1], [0, 1, -1, -1, -1], [1, 1, -1, -1, -1], [-1, -1, -1, -1, -1]],
        [[0, 1, -1, -1, -1], [0, 1, -1, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, -1], [0, 1, -1, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, -1], [0, 1, 2, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -1, 2, 1], [0, 1, 2, -1, -1], [1, 1, 1, 1, 1], [-1, 1, 0, 0, 0]],
        [[0, 1, -2, 2, 1], [0, 1, 2, -2, 1], [1, 1, 1, 1, 1], [-2, 1, 0, 0, 0]],
    ]

    expected_rewards = [REWARD_CLEAR] * (len(expected_boards) - 1) + [REWARD_LOSE]
    expected_dones = [False] * (len(expected_boards) - 1) + [True]

    assert_game(ms_game, actions, expected_boards, expected_rewards, expected_dones)


def test_reset_and_reseed():
    """Tests resetting the game and re-seeding."""

    size = TEST_BOARD_SIZE
    ms_game = create_game()

    test_win(ms_game)
    ms_game.reset()
    ms_game.seed(TEST_SEED)  # need to re-seed so it's deterministic

    test_lose(ms_game)
    ms_game.reset()

    assert ms_game.get_status() is None
    assert ms_game.hist == []
    npt.assert_array_equal(ms_game.board_size, (4, 5))
    assert ms_game.num_mines == TEST_NUM_MINES

    expected_board = [[SPACE_UNKNOWN] * size[1]] * size[0]
    npt.assert_array_equal(ms_game.board, expected_board)


def test_render():
    """Tests game rendering"""

    # get losing board
    ms_game = create_game()
    test_lose(ms_game)

    class WriteSideEffect:
        """Mock class for writable classes."""
        out = ""

        def write(self, text):
            """Appends text to internal buffer."""
            self.out += str(text)

        def get(self):
            """Gets the internal buffer."""
            return self.out

    expected_board = "0 1 X 2 1\n" \
                     "0 1 2 X 1\n" \
                     "1 1 1 1 1\n" \
                     "X 1 0 0 0"

    human_se = WriteSideEffect()
    with patch("sys.stdout.write", side_effect=human_se.write):
        ms_game.render('human')
        assert human_se.get() == expected_board

    string_io = ms_game.render('ansi')
    string_io.seek(0)
    assert string_io.read() == expected_board

    img = ms_game.render('rgb_array')
    expected_img = np.array(Image.open("images/test/render.bmp"))[:, :, :3]
    npt.assert_array_equal(img, expected_img)

    pytest.raises(NotImplementedError, ms_game.render, 'other')


def test_get_possible_moves():
    """Asserts that get_possible_moves returns only unknown spaces, or None if the game is over"""

    ms_game = create_game()
    npt.assert_array_equal(
        np.sort(ms_game.get_possible_moves(), axis=0),
        np.sort(np.transpose([
            np.tile(range(TEST_BOARD_SIZE[0]), TEST_BOARD_SIZE[1]),
            np.repeat(range(TEST_BOARD_SIZE[1]), TEST_BOARD_SIZE[0])
        ]),
                axis=0))

    ms_game.step((0, 0))
    npt.assert_array_equal(
        np.sort(ms_game.get_possible_moves(), axis=0),
        np.sort([(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3),
                 (3, 4)],
                axis=0))

    ms_game.reset()
    ms_game.seed(TEST_SEED)
    test_win(ms_game)
    assert ms_game.get_possible_moves() is None

    ms_game.reset()
    ms_game.seed(TEST_SEED)
    test_lose(ms_game)
    assert ms_game.get_possible_moves() is None
