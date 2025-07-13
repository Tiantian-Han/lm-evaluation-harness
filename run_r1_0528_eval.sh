#!/bin/bash

# ==============================================================================
# DeepSeek-R1-0528 MMLU Pro Evaluation Script
# ==============================================================================
#
# This script runs the MMLU Pro evaluation for the DeepSeek-R1-0528 model
# using vLLM for optimized inference.
#
# Key features enabled by this script:
# 1.  System Prompt: Uses the recommended system prompt for R1-0528.
# 2.  Math-Specific Prompt: Automatically applies special instructions for math problems.
# 3.  Temperature Control: Sets temperature to 0.6 as recommended.
# 4.  Correct Model Detection: Sets environment variables so the framework
#     correctly identifies the model as DeepSeek-R1-0528.
#
# ==============================================================================

# --- Configuration ---
# Set environment variables for the evaluation framework to detect the model type
export MODEL_PATH="/mnt/yrfs/llm_weights/DeepSeek-R1-0528-Qwen3-8B"
export PRETRAINED_MODEL="/mnt/yrfs/llm_weights/DeepSeek-R1-0528-Qwen3-8B"

# Activate Conda environment
# /source activate llama3_mmlu

# --- Task List ---
# List of all 14 MMLU Pro tasks for the R1-0528 model
TASKS=(
    "mmlu_pro_r1_0528_biology"
    "mmlu_pro_r1_0528_business"
    "mmlu_pro_r1_0528_chemistry"
    "mmlu_pro_r1_0528_computer_science"
    "mmlu_pro_r1_0528_economics"
    "mmlu_pro_r1_0528_engineering"
    "mmlu_pro_r1_0528_health"
    "mmlu_pro_r1_0528_history"
    "mmlu_pro_r1_0528_law"
    "mmlu_pro_r1_0528_math"
    "mmlu_pro_r1_0528_other"
    "mmlu_pro_r1_0528_philosophy"
    "mmlu_pro_r1_0528_physics"
    "mmlu_pro_r1_0528_psychology"
)

# Join tasks with a comma
TASK_STRING=$(IFS=,; echo "${TASKS[*]}")

# --- Run Evaluation ---
echo "Starting MMLU Pro evaluation for DeepSeek-R1-0528..."
echo "Model: ${MODEL_PATH}"
echo "Tasks: ${TASK_STRING}"

CUDA_VISIBLE_DEVICES="0,1,2,3" lm_eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=4,trust_remote_code=True \
    --tasks ${TASK_STRING} \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/mmlu_pro_r1_0528_results.json

echo "Evaluation finished. Results saved to results/mmlu_pro_r1_0528_results.json"
