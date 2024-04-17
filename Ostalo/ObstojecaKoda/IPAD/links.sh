#!/usr/bin/env bash

DATASETS="${1:-/hdd/IPAD/Datasets}"
RESULTS="${2:-"${DATASETS/Datasets/Results}"}"

# Change to current script directory and restore old dir on exit
pushd "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}" || realpath "${BASH_SOURCE[0]}")")" >/dev/null
trap 'popd >/dev/null' EXIT

# Datasets
for DATASET in "$DATASETS"/eyes*; do
	[ -e "$(basename "$DATASET")" ] || ln -s "$DATASET"
done

# Results
[ -e logs ] || ln -s "$RESULTS" logs
[ -e test ] || { mkdir -p "$RESULTS/test" && ln -s "$RESULTS/test"; }

# UNet teachers (needed for distillation and LeGR)
for RESULT in "$RESULTS"/*_unet; do
	mkdir -p "${RESULT}/teacher/models"
	for MODEL in "best" "final"; do
		[ -e "${RESULT}/teacher/models/teacher_${MODEL}.pkl" ] || ln "${RESULT}/original/models/unet_${MODEL}.pkl" "${RESULT}/teacher/models/teacher_${MODEL}.pkl"
	done
done

if [ -d "../LeGR" ]; then
	# LeGR model code (needed for loading LeGR models)
	[ -e model ] || ln -s "../LeGR/model"
	# LeGR model checkpoint folder
	[ -e "../LeGR/ckpt" ] || ln -s "${RESULTS}/LeGR" "../LeGR/ckpt"
fi