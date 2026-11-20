#!/bin/bash
#SBATCH --job-name=eco-qwen7b-v4-holdout
#SBATCH --comment="Final v4 frozen regression, only after full development gate"
#SBATCH --partition=A800
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:a800:1

set -euo pipefail
cd /share/home/u24121/ecocompanion_qwen_v1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
python -c "import json,hashlib; from pathlib import Path; assert hashlib.sha256(Path('training/prepared/test.jsonl').read_bytes()).hexdigest()=='8bb987af81de8c4d6f758ffca58d5f120ea96dd29415b5ca6350ceb1a1688e08'; r=json.loads(Path('results/qwen-evidence-v4-validation.json').read_text()); assert r['valid']==r['grounded']==r['safety_match']==r['total'], 'Development gate failed: frozen regression will not run'"
python scripts/evaluate_adapter.py --base models/qwen2.5-7b-instruct --adapter runs/qwen-evidence-v4/adapter --data training/prepared --split test --output results/qwen-evidence-v4-heldout-transformers.json
echo "V4_FROZEN_REGRESSION_COMPLETE"
