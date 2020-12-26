#!/usr/bin/env bash

set -eo pipefail

pylint --rcfile=setup.cfg gym_minesweeper setup.py
shellcheck scripts/hooks/* scripts/*.sh
