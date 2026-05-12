#!/bin/bash

set -e

# --- Configuration Grid ---
Ls=(8 16 32)
Cs=(256)
N_MAXs=(512)

MODES=("learnable")
NORMS=(1)
KERNELS=("rbf")
NORMDECAYS=(1)

# --- Fixed Parameters ---
TASK="rbfmix"
D=16
N_MODE="mixture"
N_MIN=64
SEED=0
BATCH_SIZE=128

RBFMIX_ELL_GRID=(0.4 0.8 1.2)
RBFMIX_SIGMA_GRID=(0.1 0.2 0.3)

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

                            # Keep folder naming consistent with train.py for rbfmix
                            if [ "$CURR_KERNEL" == "softmax" ]; then
                                KERNEL_TAG="_softmax"
                            else
                                KERNEL_TAG=""
                            fi

                            # --- Auto-generate ellmix and sigmix tags from input arrays ---
                            ELL_STR=$(printf "%s-" "${RBFMIX_ELL_GRID[@]}")
                            ELL_STR=${ELL_STR%-}
                            ELLMIX_TAG="_ellmix${ELL_STR}"

                            SIG_STR=$(printf "%s-" "${RBFMIX_SIGMA_GRID[@]}")
                            SIG_STR=${SIG_STR%-}
                            SIGMIX_TAG="_sigmix${SIG_STR}"

                            DIR_NAME="runs/${TASK}_d${D}${ELLMIX_TAG}${SIGMIX_TAG}_C${C}_L${L}_${N_MODE}_n${N_MIN}-${N_MAX}_${MODE}${NORM_TAG}${ND_TAG}${KERNEL_TAG}_seed${SEED}"

                            echo "------------------------------------------------------------------"
                            echo "RUNNING: task=$TASK L=$L N_max=$N_MAX K=$CURR_KERNEL Mode=$MODE LR=$LR Steps=$STEPS"
                            echo "ell_grid=${RBFMIX_ELL_GRID[*]}"
                            echo "sigma_grid=${RBFMIX_SIGMA_GRID[*]}"
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
                                --rbfmix_ell_grid "${RBFMIX_ELL_GRID[@]}" \
                                --rbfmix_sigma_grid "${RBFMIX_SIGMA_GRID[@]}" \
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