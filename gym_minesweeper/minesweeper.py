"""OpenAI gym environment for minesweeper."""

import sys
from io import StringIO

import gym
import numpy as np
from PIL import Image
from gym import spaces
from gym.utils import seeding

DEFAULT_BOARD_SIZE = (16, 16)
DEFAULT_NUM_MINES = 40

SPACE_MINE = -2
SPACE_UNKNOWN = -1
SPACE_MAX = 8

REWARD_WIN = 1000
REWARD_LOSE = -100
REWARD_CLEAR = 5


def get_image_rbg_arrays():
    """Returns a list of (x, y, 3) np.arrays for all space images, indexed by space number."""
    filenames = list(range(SPACE_MAX)) + ["mine", "unknown"]
    return [np.array(Image.open("images/{}.bmp".format(filename)))[:, :, :3] for filename in filenames]


IMAGE_RBG_ARRAYS = get_image_rbg_arrays()


# Based on https://github.com/genyrosk/gym-chess/blob/master/gym_chess/envs/chess.py
# pylint: disable=R0902
class MinesweeperEnv(gym.Env):
    """Minesweeper gym environment."""

    metadata = {"render.modes": ["ansi", "human", "rgb_array"]}

    def __init__(self, board_size=DEFAULT_BOARD_SIZE, num_mines=DEFAULT_NUM_MINES):
        assert np.prod(board_size) >= num_mines
        assert len(board_size) == 2
        self.board_size, self.num_mines = board_size, num_mines
        self.hist, self.board, self._board, self._rng = None, None, None, None

        self.observation_space = spaces.Box(SPACE_MINE, SPACE_MAX + 1, board_size, np.int)
        self.action_space = spaces.Discrete(np.prod(board_size))
        self.seed()
        self.reset()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (np.array): [x, y] coordinate pair of space to clear

        Returns:
            observation (np.array[np.array]): current board state
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): currently contains nothing
        """

        target_x, target_y = tuple(action)
        assert self._is_clearable_space(target_x, target_y), "Invalid action: {}".format(action)

        # If first step, populate board
        # We do this here so that the first move never triggers a mine to explode
        if len(self.hist) == 0:
            # Place mines in private board
            mines_placed = 0
            while mines_placed < self.num_mines:
                mine_indices = list(
                    zip(*
                        [self._rng.randint(0, dim_size, self.num_mines - mines_placed)
                         for dim_size in self.board_size]))
                for i in mine_indices:
                    if self._board[i] == SPACE_UNKNOWN:
                        # prohibit mines adjacent or equal to target on first step
                        if i[0] > target_x + 1 or i[0] < target_x - 1 or i[1] > target_y + 1 or i[1] < target_y - 1:
                            self._board[i] = SPACE_MINE
                            mines_placed += 1

            # Calculate nearby mines in private board
            for calc_x in range(self.board_size[0]):
                for calc_y in range(self.board_size[1]):
                    if self._board[calc_x, calc_y] == SPACE_UNKNOWN:
                        self._board[calc_x, calc_y] = self._num_nearby_mines(calc_x, calc_y)

        self._clear_space(target_x, target_y)

        status = self.get_status()

        if status is None:
            return self.board, 5, False, dict()
        if status:
            # if won, no need to reveal mines
            return self.board, 1000, True, dict()
        # if lost, reveal mines
        self.board = self._board
        return self.board, -100, True, dict()

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function does not reset the environment's random
        number generator(s); random variables in the environment's state are
        sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` yields an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (np.array[np.array]): current board state (all unknown)
        """

        self.hist = []
        self._board = np.full(self.board_size, SPACE_UNKNOWN, np.int)
        self.board = np.array(self._board)
        return self.board

    def render(self, mode='human'):
        """Renders the environment.

        If mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a StringIO.StringIO containing a
          terminal-style text representation. The text may include newlines
          and ANSI escape sequences (e.g. for colors).

        Args:
            mode (str): the mode to render with

        Returns:
            out (StringIO/None/np.ndarray):
                StringIO stream if mode is ansi
                None if mode is human
                numpy.ndarray with shape (x, y, 3) if mode is rgb_array
        """

        if mode == 'rgb_array':
            full = None
            for dim_1 in self.board:
                col = None
                for dim_2 in dim_1:
                    img = IMAGE_RBG_ARRAYS[dim_2]
                    col = img if col is None else np.concatenate((col, img), axis=1)
                full = col if full is None else np.concatenate((full, col), axis=0)
            return full

        outfile = StringIO() if mode == 'ansi' else sys.stdout if mode == 'human' else super().render(mode)
        for i, dim_1 in enumerate(self.board):
            for j, dim_2 in enumerate(dim_1):
                if dim_2 == SPACE_MINE:
                    outfile.write('X')
                elif dim_2 == SPACE_UNKNOWN:
                    outfile.write('-')
                else:
                    outfile.write(str(dim_2))
                if j != self.board_size[1] - 1:
                    outfile.write(' ')
            if i != self.board_size[0] - 1:
                outfile.write('\n')
        if mode == 'ansi':
            return outfile
        return None

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. In this case, the length is 1.
        """

        self._rng, seed = seeding.np_random(seed)
        return [seed]

    def _is_valid_space(self, target_x, target_y):
        return 0 <= target_x < self.board_size[0] and 0 <= target_y < self.board_size[1]

    def _is_clearable_space(self, target_x, target_y):
        return self._is_valid_space(target_x, target_y) and self.board[target_x, target_y] == SPACE_UNKNOWN

    def _num_nearby_mines(self, target_x, target_y):
        num_mines = 0
        for i in range(target_x - 1, target_x + 2):
            for j in range(target_y - 1, target_y + 2):
                if (target_x != i or target_y != j) and self._is_valid_space(i, j) and self._board[i, j] == SPACE_MINE:
                    num_mines += 1
        return num_mines

    def _clear_space(self, target_x, target_y):
        spaces_to_clear = {(target_x, target_y)}
        spaces_cleared = set()

        update_hist = True
        while spaces_to_clear:
            current_space = next(iter(spaces_to_clear))
            self.board[current_space[0], current_space[1]] = self._board[current_space[0], current_space[1]]
            if update_hist:
                self.hist.append(current_space)
                update_hist = False

            spaces_to_clear.remove(current_space)
            spaces_cleared.add(current_space)

            if self.board[current_space[0], current_space[1]] == 0:
                for i in range(current_space[0] - 1, current_space[0] + 2):
                    for j in range(current_space[1] - 1, current_space[1] + 2):
                        if self._is_valid_space(i, j) and (i, j) not in spaces_cleared:
                            spaces_to_clear.add((i, j))

    def get_status(self):
        """Gets the status of the game.

        Returns:
            status (bool): True if game won, False if game lost, None if game in progress
        """

        if np.count_nonzero(self.board == SPACE_MINE):
            return False
        return True if np.count_nonzero(self.board == SPACE_UNKNOWN) == self.num_mines else None

    def get_possible_moves(self):
        """Gets a collection of possible moves.

        Returns:
            moves (list): List of (x, y) pairs that are possible moves. If the game is over, returns None.
        """

        if self.get_status() is None:
            all_coords = np.transpose([
                np.tile(range(self.board_size[0]), self.board_size[1]),
                np.repeat(range(self.board_size[1]), self.board_size[0])
            ])
            return [tuple(coord) for coord in all_coords if self._is_clearable_space(*tuple(coord))]
        return None
