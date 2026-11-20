#!/bin/bash
#SBATCH --job-name=eco-qwen7b-v2
#SBATCH --comment="Diagnostic longer synthetic evidence QLoRA; held-out test excluded"
#SBATCH --partition=L40
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1

set -euo pipefail
cd /share/home/u24121/ecocompanion_qwen_v1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

python scripts/train_qwen_qlora.py \
  --model models/qwen2.5-7b-instruct \
  --data training/prepared \
  --epochs 10 \
  --logging-steps 1 \
  --max-length 768 \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --output runs/qwen-evidence-v2
python scripts/evaluate_adapter.py \
  --base models/qwen2.5-7b-instruct \
  --adapter runs/qwen-evidence-v2/adapter \
  --data training/prepared \
  --output results/qwen-evidence-v2-validation.json
echo "V2_TRAINING_AND_VALIDATION_COMPLETE"
