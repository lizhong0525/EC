#!/bin/bash
#SBATCH --job-name=eco-qwen-validate-base
#SBATCH --comment="Verify full official Qwen 7B base before training"
#SBATCH --partition=intel
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
set -euo pipefail
cd /share/home/u24121/ecocompanion_qwen_v1
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -c "import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; p='models/qwen2.5-7b-instruct'; t=AutoTokenizer.from_pretrained(p, local_files_only=True); m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, local_files_only=True); print(type(m).__name__, sum(x.numel() for x in m.parameters()), len(t))"
echo PREPARATION_COMPLETE
