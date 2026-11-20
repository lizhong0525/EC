#!/bin/bash
#SBATCH --job-name=eco-qwen7b-v1
#SBATCH --comment="EcoCompanion synthetic evidence QLoRA v1; not clinical validation"
#SBATCH --partition=L40
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1

set -euo pipefail
ROOT=/share/home/u24121/ecocompanion_qwen_v1
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

nvidia-smi
python -c "import torch, bitsandbytes; print('torch', torch.__version__, 'CUDA', torch.version.cuda); assert torch.__version__.startswith('2.7.1'); assert torch.cuda.is_available()"
python scripts/train_qwen_qlora.py \
  --model models/qwen2.5-7b-instruct \
  --data training/prepared \
  --epochs 2 \
  --max-length 768 \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --output runs/qwen-evidence-v1
python scripts/evaluate_adapter.py \
  --base models/qwen2.5-7b-instruct \
  --adapter runs/qwen-evidence-v1/adapter \
  --data training/prepared \
  --output results/qwen-evidence-v1-validation.json
python scripts/package_adapter.py \
  --run runs/qwen-evidence-v1 \
  --data training/prepared \
  --validation results/qwen-evidence-v1-validation.json \
  --output deliverables/eco-qwen-evidence-v1.tar.gz
echo "TRAINING_AND_PACKAGE_COMPLETE"
