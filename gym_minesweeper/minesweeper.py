"""OpenAI gym environment for minesweeper."""

import sys
from io import StringIO

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

DEFAULT_BOARD_SIZE = (16, 30)
DEFAULT_NUM_MINES = 99

SPACE_MINE = -2
SPACE_UNKNOWN = -1
SPACE_MAX = 8

REWARD_WIN = 1000
REWARD_LOSE = -100
REWARD_CLEAR = 5


# Based on https://github.com/genyrosk/gym-chess/blob/master/gym_chess/envs/chess.py
# pylint: disable=R0902
class MinesweeperEnv(gym.Env):
    """Minesweeper gym environment."""

    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, board_size=DEFAULT_BOARD_SIZE, num_mines=DEFAULT_NUM_MINES):
        assert np.prod(board_size) >= num_mines
        assert len(board_size) == 2
        self.board_size, self.num_mines = board_size, num_mines
        self.hist, self.board, self._board, self._rng = None, None, None, None

        self.observation_space = spaces.Box(SPACE_MINE, SPACE_MAX + 1, board_size, np.int)
        self.action_space = spaces.Discrete(np.prod(board_size))
        self.reset()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (np.array): [x, y] coordinate pair of space to clear

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
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

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """

        self.hist = []
        self._board = np.full(self.board_size, SPACE_UNKNOWN, np.int)
        self.board = np.array(self._board)
        return self.board

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        for dim_1 in self.board:
            for dim_2 in dim_1:
                if dim_2 == SPACE_MINE:
                    outfile.write('X')
                elif dim_2 == SPACE_UNKNOWN:
                    outfile.write(' ')
                else:
                    outfile.write(dim_2)
                outfile.write(' ')
            outfile.write('\n')
        return outfile

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
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
