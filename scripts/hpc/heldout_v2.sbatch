#!/bin/bash
#SBATCH --job-name=eco-qwen7b-heldout
#SBATCH --comment="One frozen held-out base and v2 comparison; no further tuning from test"
#SBATCH --partition=L40
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1

set -euo pipefail
cd /share/home/u24121/ecocompanion_qwen_v1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

python scripts/evaluate_adapter.py \
  --base models/qwen2.5-7b-instruct \
  --data training/prepared --split test \
  --output results/qwen-evidence-base-heldout-transformers.json
python scripts/evaluate_adapter.py \
  --base models/qwen2.5-7b-instruct \
  --adapter runs/qwen-evidence-v2/adapter \
  --data training/prepared --split test \
  --output results/qwen-evidence-v2-heldout-transformers.json
echo "FROZEN_COMPARISON_COMPLETE"
