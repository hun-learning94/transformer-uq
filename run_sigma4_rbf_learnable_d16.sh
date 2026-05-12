#!/bin/bash

set -e

# --- Configuration Grid ---
Ls=(32 16 8)
Cs=(256)
N_MAXs=(256)
SIGMAS=(0.4)

MODES=("learnable")
NORMS=(1)
KERNELS=("rbf")
NORMDECAYS=(1)

# --- Fixed Parameters ---
TASK="rbf"
D=16
N_MODE="mixture"
N_MIN=64
SEED=0
BATCH_SIZE=128
ELL=1.6

for SIGMA in "${SIGMAS[@]}"; do
    for L in "${Ls[@]}"; do
        for C in "${Cs[@]}"; do
            for N_MAX in "${N_MAXs[@]}"; do
                for KERNEL in "${KERNELS[@]}"; do
                    for MODE in "${MODES[@]}"; do
                        for NORM in "${NORMS[@]}"; do
                            for NORMDECAY in "${NORMDECAYS[@]}"; do

                                CURR_NORM=$NORM
                                CURR_KERNEL=$KERNEL
                                CURR_NORMDECAY=$NORMDECAY

                                if [ "$CURR_NORMDECAY" -eq 1 ]; then
                                    if [ "$CURR_NORM" -ne 1 ]; then CURR_NORM=1; fi
                                fi

                                if [ "$CURR_KERNEL" == "softmax" ]; then
                                    if [ "$CURR_NORM" -eq 0 ]; then CURR_NORM=1; fi
                                fi

                                if [ "$TASK" == "blr" ]; then
                                    if [ "$CURR_KERNEL" != "auto" ]; then CURR_KERNEL="auto"; fi
                                fi

                                if [ "$MODE" == "theory" ]; then
                                    LR=1e-3
                                    STEPS=50000
                                fi
                                if [ "$MODE" == "learnable" ]; then
                                    LR=2e-4
                                    STEPS=100000
                                fi

                                if [ "$CURR_NORM" -eq 1 ]; then NORM_TAG="_norm"; else NORM_TAG=""; fi
                                if [ "$CURR_NORMDECAY" -eq 1 ]; then ND_TAG="_nd"; else ND_TAG=""; fi

                                # Keep folder naming compatible with your older RBF runs:
                                # no explicit _rbf suffix, but do include sigma.
                                if [ "$CURR_KERNEL" == "softmax" ]; then
                                    KERNEL_TAG="_softmax"
                                else
                                    KERNEL_TAG=""
                                fi

                                SIG_TAG="_sig${SIGMA}"
                                ELL_TAG="_ell${ELL}"

                                DIR_NAME="runs/${TASK}_d${D}${ELL_TAG}${SIG_TAG}_C${C}_L${L}_${N_MODE}_n${N_MIN}-${N_MAX}_${MODE}${NORM_TAG}${ND_TAG}${KERNEL_TAG}_seed${SEED}"

                                echo "------------------------------------------------------------------"
                                echo "RUNNING: sigma=$SIGMA ell=$ELL L=$L N_max=$N_MAX K=$CURR_KERNEL Mode=$MODE LR=$LR Steps=$STEPS"
                                echo "save_dir: $DIR_NAME"
                                echo "------------------------------------------------------------------"

                                RESUME_ARG=""

                                LATEST_STEP_CKPT=$(find "$DIR_NAME" -maxdepth 1 -type f -name 'ckpt_step_*.pt' | sort -V | tail -n 1)

                                if [ -n "$LATEST_STEP_CKPT" ]; then
                                    if [[ "$LATEST_STEP_CKPT" =~ ckpt_step_([0-9]+)\.pt$ ]]; then
                                        FOUND_STEP="${BASH_REMATCH[1]}"
                                    else
                                        FOUND_STEP=0
                                    fi

                                    echo "Found checkpoint at step: $FOUND_STEP (Target: $STEPS)"

                                    if [ "$FOUND_STEP" -lt "$STEPS" ]; then
                                        echo "Resuming run (step $FOUND_STEP < $STEPS)..."
                                        RESUME_ARG="--resume $LATEST_STEP_CKPT"
                                    else
                                        echo "Checkpoint reached target step. Skipping."
                                        continue
                                    fi
                                else
                                    echo "No step checkpoint found. Starting fresh."
                                fi

                                python train.py \
                                    --task $TASK \
                                    --kernel $CURR_KERNEL \
                                    --mode $MODE \
                                    --normalize $CURR_NORM \
                                    --normdecay $CURR_NORMDECAY \
                                    --d $D \
                                    --n_mode $N_MODE \
                                    --n_min $N_MIN \
                                    --n_max $N_MAX \
                                    --L $L \
                                    --C $C \
                                    --rbf_ell $ELL \
                                    --sigma $SIGMA \
                                    --seed $SEED \
                                    --batch $BATCH_SIZE \
                                    --save_every 500 \
                                    --save_best 1 \
                                    --lr $LR \
                                    --steps $STEPS \
                                    $RESUME_ARG

                                echo ""
                            done
                        done
                    done
                done
            done
        done
    done
done