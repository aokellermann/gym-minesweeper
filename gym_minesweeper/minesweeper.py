import sys
from io import StringIO

import gym
from gym import spaces
import numpy as np

DEFAULT_BOARD_SIZE = (16, 30)
DEFAULT_NUM_MINES = 99

SPACE_MINE = -2
SPACE_UNKNOWN = -1
SPACE_MAX = 8


# Based on https://github.com/genyrosk/gym-chess/blob/master/gym_chess/envs/chess.py
class MinesweeperEnv(gym.Env):
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, board_size=DEFAULT_BOARD_SIZE, num_mines=DEFAULT_NUM_MINES):
        assert np.prod(board_size) >= num_mines
        assert len(board_size) == 2
        self.board_size, self.num_mines = board_size, num_mines
        self.hist, self.board, self._board = None, None, None

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

        x, y = tuple(action)
        assert self._is_valid_space(x, y) and self._board[x, y] < 0, "Invalid action: {}".format(action)

        # If already cleared, admonish user
        if self.board[x, y] >= 0:
            return self.board, -1, False, dict()

        self._clear_space(x, y)

        status = self.get_status()

        if status is None:
            return self.board, 5, False, dict()
        elif status:
            return self.board, 1000, True, {"master": self._board}
        else:
            return self.board, -100, True, {"master": self._board}

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

        # Private board -- contains all mine information
        self._board = np.full(self.board_size, SPACE_UNKNOWN, np.int)
        # Public board
        self.board = np.array(self._board)

        # Place mines in private board
        mines_placed = 0
        while mines_placed < self.num_mines:
            mine_indices = list(
                zip(*[np.random.randint(0, dim_size, self.num_mines - mines_placed) for dim_size in self.board_size]))
            for i in range(self.num_mines):
                if self._board[mine_indices] == SPACE_UNKNOWN:
                    self._board[mine_indices] = SPACE_MINE
                    mines_placed += 1

        # Calculate nearby mines in private board
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if self._board[x] == SPACE_UNKNOWN:
                    self._board[x] = self._num_nearby_mines(x, y)

        self.hist = []

        self.hist, self.board, self._board = None, None, None
        return np.full(self.board_size, SPACE_UNKNOWN, np.int)

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
        for d1 in self.board:
            for d2 in d1:
                if d2 == SPACE_MINE:
                    outfile.write('X')
                elif d2 == SPACE_UNKNOWN:
                    outfile.write(' ')
                else:
                    outfile.write(d2)
                outfile.write(' ')
            outfile.write('\n')
        return outfile

    def _is_valid_space(self, x, y):
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _num_nearby_mines(self, x, y):
        num_mines = 0
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if x != i and y != j and self._is_valid_space(i, j) and self._board[i, j] == SPACE_MINE:
                    num_mines += 1
        return num_mines

    def _clear_space(self, x, y):
        if self._is_valid_space(x, y):
            self.board[x, y] = self._board[x, y]
            self.hist.append((x, y))
            if self.board[x, y] == 0:
                for i in range(x - 1, x + 2):
                    for j in range(y - 1, y + 2):
                        if x != i and y != j and self._is_valid_space(i, j):
                            self._clear_space(i, j)

    def get_status(self):
        """Gets the status of the game.

        Returns:
            status (bool): True if game won, False if game lost, None if game in progress
        """
        if np.count_nonzero(self.board == SPACE_MINE):
            return False
        return True if np.count(self.board == SPACE_UNKNOWN) == 0 else None
