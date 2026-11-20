#!/bin/bash
#SBATCH --job-name=eco-qwen-prepare
#SBATCH --comment="EcoCompanion Qwen evidence v1: isolated environment and official base"
#SBATCH --partition=intel
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail
ROOT=/share/home/u24121/ecocompanion_qwen_v1
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Reuse the existing CUDA/PyTorch stack. Only this task's small training
# dependencies are installed; no other project's environment is modified.
uv venv --clear --system-site-packages \
  --python /share/home/u24121/.conda/envs/qwen_train/bin/python .venv
echo /share/home/u24121/.conda/envs/qwen_train/lib/python3.10/site-packages > \
  .venv/lib/python3.10/site-packages/conda-qwen-train.pth
uv pip install --python .venv/bin/python --no-deps \
  'bitsandbytes @ https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/19/57/3443d6f183436fbdaf5000aac332c4d5ddb056665d459244a5608e98ae92/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl'
source .venv/bin/activate

python -c "import torch, transformers, peft, bitsandbytes, datasets; print(torch.__version__, transformers.__version__, peft.__version__, bitsandbytes.__version__, datasets.__version__)"
python scripts/prepare_qwen_training.py
if [[ ! -f models/qwen2.5-7b-instruct/SOURCE.json ]]; then
  bash scripts/download_qwen_base.sh
fi

python -c "import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; p='models/qwen2.5-7b-instruct'; t=AutoTokenizer.from_pretrained(p, local_files_only=True); m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, local_files_only=True); print(type(m).__name__, sum(x.numel() for x in m.parameters()), len(t))"
echo "PREPARATION_COMPLETE"
