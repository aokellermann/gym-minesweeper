#!/usr/bin/env bash

set -eo pipefail

pytest --junitxml=test_results/gym_minesweeper/report.xml gym_minesweeper/tests
