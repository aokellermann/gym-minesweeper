#!/usr/bin/env bash

set -eo pipefail

if [ "$#" -eq 0 ]; then
	yapf -i -r gym_minesweeper 2>/dev/null
elif [ "$1" == "check" ]; then
	yapf -d -r gym_minesweeper
fi
