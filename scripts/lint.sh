#!/usr/bin/env bash

set -eo pipefail

pipenv run lint
shellcheck scripts/hooks/* scripts/*.sh
