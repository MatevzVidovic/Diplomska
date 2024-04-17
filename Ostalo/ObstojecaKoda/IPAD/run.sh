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
PRUNING_BS=2
PRUNED_BS=2
OMEGAS=(0 0.1 0.5 0.9 1)
PERCENTS=("" 50 25)

# Overwrite above values from run.cfg if it exists
[ -f run.cfg ] && . run.cfg

# If percent-specific batch sizes weren't defined, use existing values
[ "$PRUNED_BS_50" ] || PRUNED_BS_50=$PRUNED_BS
[ "$PRUNED_BS_25" ] || PRUNED_BS_25=$PRUNED_BS_50

# If EXTRAS wasn't defined, use default values, which differ per model
if [ -z "${EXTRAS+x}" ]; then
	[ "$MODEL" == "densenet" ] && EXTRAS=("" "--prune conv" "--channelsUseWeightsOnly") || EXTRAS=("")
fi

# Start visdom server if not already running
pgrep -f visdom.server >/dev/null || tmux new -d -s visdom "${PYTHON} -m visdom.server"



#######################################################
###          EXPERIMENTS WITH MY CRITERION          ###
#######################################################

for DATASET in "${DATASETS[@]}"; do
	for OMEGA in "${OMEGAS[@]}"; do
		for EXTRA in "${EXTRAS[@]}"; do
			EXP="${3:+$3_}"
			if [ "$NORM" != "2" ]; then
				EXP="${EXP}l${NORM}_"
			fi
			case $EXTRA in
				"--prune conv")
					EXP="${EXP}3x3_only_"
					;;
				"--channelsUseWeightsOnly")
					EXP="${EXP}1x1_weights_only_"
					;;
			esac

			PRUNING_EXP="matej_${DATASET}_${MODEL}/${EXP}pruning_omega${OMEGA}"
			if [ -d "logs/${PRUNING_EXP}" ]; then
				echo "${PRUNING_EXP} already exists, skipping"
			else
				"$PYTHON" train_with_pruning_combined.py \
					--dataset "eyes_${DATASET}" \
					--model "$MODEL" \
					--bs $PRUNING_BS \
					--workers $WORKERS \
					--norm $NORM \
					--expname "$PRUNING_EXP" \
					--resume "logs/matej_${DATASET}_${MODEL}/original/models/${MODEL}_best.pkl" \
					${EXTRA} ||
				{ EXIT=$?; rm -r "logs/${PRUNING_EXP}"; exit $EXIT; }  # Exit if there was an error
			fi

			for PERCENT in "${PERCENTS[@]}"; do
				FINAL_EXP="matej_${DATASET}_${MODEL}/${EXP}final_omega${OMEGA}${PERCENT:+_pruned$PERCENT}"
				if [ -d "logs/${FINAL_EXP}" ]; then
					echo "${FINAL_EXP} already exists, skipping"
					continue
				fi
				BSVAR="PRUNED_BS${PERCENT:+_$PERCENT}"
				"$PYTHON" train_pruned_model.py \
					--dataset "eyes_${DATASET}" \
					--model "$MODEL" \
					--bs ${!BSVAR} \
					--workers $WORKERS \
					--expname "$FINAL_EXP" \
					--resume "logs/${PRUNING_EXP}/models/${MODEL}_${PERCENT:-final}.pt" ||
				{ EXIT=$?; rm -r "logs/${FINAL_EXP}"; exit $EXIT; }  # Exit if there was an error
			done
		done
	done
done



#######################################################
###                    BASELINES                    ###
#######################################################

# for DATASET in "${DATASETS[@]}"; do
# 	if [ "$NORM" == "2" ]; then
# 		EXTRA="uniform"
# 	else
# 		EXTRA="random"
# 	fi
# 	PRUNING_EXP="matej_${DATASET}_${MODEL}/${EXTRA}_pruning"
# 	if [ -d "logs/${PRUNING_EXP}" ]; then
# 		echo "${PRUNING_EXP} already exists, skipping"
# 	else
# 		"$PYTHON" train_with_pruning_combined.py \
# 			--dataset "eyes_${DATASET}" \
# 			--model "$MODEL" \
# 			--bs $PRUNING_BS \
# 			--workers $WORKERS \
# 			--expname "$PRUNING_EXP" \
# 			--resume "logs/matej_${DATASET}_${MODEL}/original/models/${MODEL}_best.pkl" \
# 			--${EXTRA} ||
# 		{ EXIT=$?; rm -r "logs/${PRUNING_EXP}"; exit $EXIT; }  # Exit if there was an error
# 	fi

# 	for PERCENT in "${PERCENTS[@]}"; do
# 		FINAL_EXP="matej_${DATASET}_${MODEL}/${EXTRA}${PERCENT:+_pruned$PERCENT}"
# 		if [ -d "logs/${FINAL_EXP}" ]; then
# 			echo "${FINAL_EXP} already exists, skipping"
# 			continue
# 		fi
# 		BSVAR="PRUNED_BS${PERCENT:+_$PERCENT}"
# 		"$PYTHON" train_pruned_model.py \
# 			--dataset "eyes_${DATASET}" \
# 			--model "$MODEL" \
# 			--bs ${!BSVAR} \
# 			--workers $WORKERS \
# 			--expname "$FINAL_EXP" \
# 			--resume "logs/${PRUNING_EXP}/models/${MODEL}_${PERCENT:-final}.pt" ||
# 		{ EXIT=$?; rm -r "logs/${FINAL_EXP}"; exit $EXIT; }  # Exit if there was an error
# 	done
# done