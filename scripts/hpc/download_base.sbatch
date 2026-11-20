#!/bin/bash
#SBATCH --job-name=eco-qwen-base-download
#SBATCH --comment="Download and verify pinned Qwen base; no training data or credentials"
#SBATCH --partition=intel
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /share/home/u24121/ecocompanion_qwen_v1
export PYTHONUNBUFFERED=1
curl -fsSL --connect-timeout 20 --max-time 60 \
  -o /dev/null \
  "https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/a09a35458c702b33eeacc393d103063234e8bc28/config.json"
echo COMPUTE_NETWORK_OK
bash scripts/download_qwen_base.sh
