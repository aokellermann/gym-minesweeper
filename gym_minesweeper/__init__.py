"""OpenAI gym environment for minesweeper."""

__all__ = [
    'MinesweeperEnv', 'SPACE_MINE', 'SPACE_UNKNOWN', 'DEFAULT_REWARD_WIN', 'DEFAULT_REWARD_LOSE', 'DEFAULT_REWARD_CLEAR'
]

from gym.envs.registration import register

from gym_minesweeper.minesweeper import MinesweeperEnv, SPACE_MINE, SPACE_UNKNOWN, DEFAULT_REWARD_WIN, DEFAULT_REWARD_LOSE, DEFAULT_REWARD_CLEAR

register(id='Minesweeper-v0', entry_point='gym_minesweeper:MinesweeperEnv')
