# gym-minesweeper

[![aokellermann](https://circleci.com/gh/aokellermann/gym-minesweeper.svg?style=svg)](https://app.circleci.com/pipelines/github/aokellermann/gym-minesweeper)

A minesweeper environment for [OpenAI Gym](https://gym.openai.com/).

## Setup

You may have to install the following build dependencies (due to Pillow not providing a wheel on Linux):

* lcms2
* libtiff
* openjpeg2
* libimagequant
* libxcb

Then, install the environment:

```bash
pip install -e .
```

## Play

Example usage:

```python3
import random

import gym
from PIL import Image

from gym_minesweeper import SPACE_UNKNOWN, SPACE_MINE

# Creates a new game
env = gym.make("Minesweeper-v0")

# Prints the board size and num mines
print("board size: {}, num mines: {}".format(env.board_size, env.num_mines))

# Clear a random space (the first clear will never explode a mine, and there will be no nearby bombs)
move = random.choice(env.get_possible_moves())
board, reward, done, _ = env.step(move)

# Get the value of the cleared space
space = board[move]
if space >= 0:
    print("There are {} nearby mines.".format(space))
elif space == SPACE_UNKNOWN:
    print("This space isn't cleared yet.")
elif space == SPACE_MINE:
    print("You hit a mine!")

# Prints human readable board in terminal
env.render()

# Saves an image of the board
Image.fromarray(env.render('rgb_array')).save("board.jpg")

# Prints your move history, a list of coordinates
print(env.hist)
```
