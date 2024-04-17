#!/usr/bin/env bash

# Change to current script directory and restore old dir on exit
pushd "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}" || realpath "${BASH_SOURCE[0]}")")" >/dev/null
trap 'popd >/dev/null' EXIT

# Arguments
export CUDA_VISIBLE_DEVICES=${1:-0}
NORM=${2:-$((2 - $CUDA_VISIBLE_DEVICES))}

# Default values
PYTHON=~/"conda/envs/ritnet/bin/python"
MODEL="densenet"
DATASETS=("mobius_sip" "sbvpi" "smd" "sld")
WORKERS=8
LEGR_BS=2
PRUNED_BS=2
OMEGAS=(0 0.1 0.5 0.9 1)
PERCENTS=("" 50 25)

# Overwrite above values from run.cfg if it exists
[ -f "../IPAD/run.cfg" ] && . "../IPAD/run.cfg"
[ -f run.cfg ] && . run.cfg

# If percent-specific batch sizes weren't defined, use existing values
[ "$PRUNED_BS_50" ] || PRUNED_BS_50="$PRUNED_BS"
[ "$PRUNED_BS_25" ] || PRUNED_BS_25="$PRUNED_BS_50"

# Start visdom server if not already running
pgrep -f visdom.server >/dev/null || tmux new -d -s visdom "${PYTHON} -m visdom.server"

# Run experiments
for DATASET in "${DATASETS[@]}"; do
	for OMEGA in "${OMEGAS[@]}"; do
		for PERCENT in "${PERCENTS[@]}"; do
			EXP="matej_${DATASET}_${MODEL}"
			PREFIX="${3:+$3_}"
			if [ "$NORM" != "2" ]; then
				PREFIX="${PREFIX}l${NORM}_"
			fi
			SUFFIX="omega${OMEGA}${PERCENT:+_pruned$PERCENT}"
			FULLEXP="${EXP}/${PREFIX}${SUFFIX}"

			cd ../LeGR
			if [ -f "ckpt/${FULLEXP}_bestarch_init.pt" ]; then
				echo "${FULLEXP} already exists, skipping"
			else
				[ "$MODEL" == "unet" ] && PRUNER="FilterPrunerUNet" || PRUNER="FilterPrunerRitnet"
				"$PYTHON" legr.py \
					--name "$FULLEXP" \
					--dataset "eyes_${DATASET}" \
					--model "../IPAD/logs/${EXP}/teacher/models/teacher_best.pkl" \
					--batch_size $LEGR_BS \
					--workers $WORKERS \
					--omega $OMEGA \
					--prune_away ${PERCENT:-75} \
					--pruner $PRUNER \
					--rank_type "l${NORM}_combined" ||
				exit  # Exit loop if there was an error
			fi

			cd ../IPAD
			if [ -d "logs/${EXP}/legr_${PREFIX}${SUFFIX}" ]; then
				echo "${PREFIX}${SUFFIX} already exists, skipping"
				continue
			fi
			BSVAR="PRUNED_BS${PERCENT:+_$PERCENT}"
			"$PYTHON" train_pruned_model.py \
				--expname "${EXP}/legr_${PREFIX}${SUFFIX}" \
				--dataset "eyes_${DATASET}" \
				--model "$MODEL" \
				--resume "../LeGR/ckpt/${FULLEXP}_bestarch_init.pt" \
				--bs ${!BSVAR} \
				--workers $WORKERS ||
			{ EXIT=$?; rm -r "logs/${EXP}/legr_${PREFIX}${SUFFIX}"; exit $EXIT; }  # Exit loop if there was an error
		done
	done
done
