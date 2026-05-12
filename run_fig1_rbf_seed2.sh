#!/bin/bash

# Stop immediately if any command fails
set -e

# --- Configuration Grid ---
Ls=(2 4 8 16 32)
Cs=(256 128 64 32 16)
N_MAXs=(128)
MODES=("theory" "learnable")
NORMS=(0)
KERNELS=("auto")
NORMDECAYS=(0)

# --- Fixed Parameters ---
TASK="rbf"
D=2
N_MODE="mixture"
N_MIN=64
SEED=2
BATCH_SIZE=128

# Loop over L, C, and N_MAX
for L in "${Ls[@]}"; do
    for C in "${Cs[@]}"; do
        for N_MAX in "${N_MAXs[@]}"; do

            # Loop: KERNEL
            for KERNEL in "${KERNELS[@]}"; do
                for MODE in "${MODES[@]}"; do
                    for NORM in "${NORMS[@]}"; do
                        for NORMDECAY in "${NORMDECAYS[@]}"; do

                            # --- CONSTRAINT CHECKS ---
                            if [ "$NORMDECAY" -eq 1 ]; then
                                if [ "$NORM" -ne 1 ]; then NORM=1; fi
                            fi
                            if [ "$KERNEL" == "softmax" ]; then
                                if [ "$NORM" -eq 0 ]; then NORM=1; fi
                            fi
                            if [ "$TASK" == "blr" ]; then
                                if [ "$KERNEL" != "auto" ]; then KERNEL="auto"; fi
                            fi
                            # --- Hyperparameter Logic ---
                            if [ "$MODE" == "theory" ]; then
                                LR=1e-3
                                STEPS=50000; 
                            fi
                            if [ "$MODE" == "learnable" ]; then
                                LR=2e-4
                                STEPS=100000; 
                            fi
                            # --- Directory Names ---
                            if [ "$NORM" -eq 1 ]; then NORM_TAG="_norm"; else NORM_TAG=""; fi
                            if [ "$NORMDECAY" -eq 1 ]; then ND_TAG="_nd"; else ND_TAG=""; fi
                            if [ "$KERNEL" == "softmax" ]; then KERNEL_TAG="_softmax"; else KERNEL_TAG=""; fi

                            DIR_NAME="runs/${TASK}_d${D}_C${C}_L${L}_${N_MODE}_n${N_MIN}-${N_MAX}_${MODE}${NORM_TAG}${ND_TAG}${KERNEL_TAG}_seed${SEED}"

                            echo "------------------------------------------------------------------"
                            echo "RUNNING: L=$L N=$N_MAX K=$KERNEL Mode=$MODE LR=$LR Steps=$STEPS"
                            echo "------------------------------------------------------------------"

                            # --- SMART RESUME LOGIC ---
                            RESUME_ARG=""
                            
                            # Find the latest step checkpoint (sort by version number)
                            LATEST_STEP_CKPT=$(ls -v "${DIR_NAME}"/ckpt_step_*.pt 2>/dev/null | tail -n 1)
                            
                            if [ -n "$LATEST_STEP_CKPT" ]; then
                                if [[ $LATEST_STEP_CKPT =~ ckpt_step_([0-9]+).pt ]]; then
                                    FOUND_STEP="${BASH_REMATCH[1]}"
                                else
                                    FOUND_STEP=0
                                fi

                                echo "🔍 Found checkpoint at step: $FOUND_STEP (Target: $STEPS)"

                                if [ "$FOUND_STEP" -lt "$STEPS" ]; then
                                    echo "🔄 Resuming run (Step $FOUND_STEP < $STEPS)..."
                                    RESUME_ARG="--resume $LATEST_STEP_CKPT"
                                else
                                    echo "✅ Checkpoint reached target step. Skipping."
                                    continue
                                fi
                            else
                                echo "✨ No checkpoints found. Starting fresh."
                            fi

                            # --- Run Training ---
                            python train.py \
                                --task $TASK \
                                --kernel $KERNEL \
                                --mode $MODE \
                                --normalize $NORM \
                                --normdecay $NORMDECAY \
                                --d $D \
                                --n_mode $N_MODE \
                                --n_min $N_MIN \
                                --n_max $N_MAX \
                                --L $L \
                                --C $C \
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