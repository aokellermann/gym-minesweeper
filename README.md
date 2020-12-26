# gym-minesweeper

A minesweeper environment for [OpenAI Gym](https://gym.openai.com/).

## Setup

You may have to install the following dependencies (due to Pillow not providing a wheel on Linux):

* lcms2
* libtiff
* openjpeg2
* libimagequant
* libxcb

Then, to install the environment

```bash
pip install -e .
```

## Play

```python3
import gym
import gym_minesweeper
import random

from PIL import Image

# Creates a new game
env = gym.make("Minesweeper-v0")

# Prints the board size and num mines
print("board size: {}, num mines: {}".format(env.board_size, env.num_mines))

# Clear a random space. Takes in coordinates.
board, reward, done, _ = env.step((random.randint(0, env.board_size[0] - 1), random.randint(0, env.board_size[1] - 1)))

# Prints human readable board in terminal
env.render()

# Saves an image of the board
Image.fromarray(env.render('rgb_array')).save("board.jpg")

# Prints your move history, a list of coordinates
print(env.hist)
```
