#!/usr/bin/env bash

set -eo pipefail

coverage run --source gym_minesweeper -m pytest --junitxml=test_results/gym_minesweeper/report.xml gym_minesweeper/tests
