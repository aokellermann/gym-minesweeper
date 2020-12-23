#!/usr/bin/env bash

set -eo pipefail

if [ "$#" -eq 0 ]; then
	pipenv run format
	shfmt -w scripts
elif [ "$1" == "check" ]; then
	pipenv run format check
	shfmt -l -d scripts
fi
