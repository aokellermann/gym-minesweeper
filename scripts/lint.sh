#!/usr/bin/env bash

set -eo pipefail

pylint --rcfile=setup.cfg gym_minesweeper
shellcheck scripts/hooks/* scripts/*.sh
