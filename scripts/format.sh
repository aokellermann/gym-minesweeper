#!/usr/bin/env bash

set -eo pipefail

if [ "$#" -eq 0 ]; then
	yapf -i -r gym_minesweeper setup.py 2>/dev/null
	shfmt -w scripts
elif [ "$1" == "check" ]; then
	yapf -d -r gym_minesweeper setup.py
	shfmt -l -d scripts
fi
