#!/bin/bash
# Run from a login node with approved external HTTP access.
set -euo pipefail
ROOT=/share/home/u24121/ecocompanion_qwen_v1
MODEL="$ROOT/models/qwen2.5-7b-instruct"
REV=a09a35458c702b33eeacc393d103063234e8bc28
BASE="https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/resolve/master"
mkdir -p "$MODEL"
if [[ ! -f "$ROOT/models/source-tree.json" ]]; then
  curl -fsSL --retry 5 --connect-timeout 20 \
    -o "$ROOT/models/source-tree.json" \
    "https://hf-mirror.com/api/models/Qwen/Qwen2.5-7B-Instruct/tree/$REV?recursive=true&expand=false"
fi
declare -A EXPECTED_SIZES=()
while read -r name size; do
  EXPECTED_SIZES["$name"]="$size"
done < <(python - <<'PY'
import json
from pathlib import Path

tree = json.loads(Path("models/source-tree.json").read_text(encoding="utf-8"))
for entry in tree:
    print(entry["path"], entry.get("size") or entry.get("lfs", {}).get("size", 0))
PY
)
for name in config.json generation_config.json tokenizer_config.json tokenizer.json \
  vocab.json merges.txt model.safetensors.index.json \
  model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  expected_size="${EXPECTED_SIZES[$name]:-}"
  if [[ -z "$expected_size" ]]; then
    echo "Missing expected size for $name" >&2
    exit 1
  fi
  if [[ -f "$MODEL/$name" ]] && [[ "$(stat -c %s "$MODEL/$name")" == "$expected_size" ]]; then
    echo "Already complete: $name"
    continue
  fi
  echo "Downloading $name"
  rm -f "$MODEL/$name.part"
  for attempt in 1 2 3 4 5; do
    # Full-file transfers only: mirror range/resume behavior was inconsistent.
    # The expected SHA-256 is the release gate, not the transport exit status.
    if curl -fsSL --connect-timeout 20 --max-time 1800 --speed-time 60 --speed-limit 1024 \
      -o "$MODEL/$name.part" "$BASE/$name"; then
      break
    fi
    if [[ -f "$MODEL/$name.part" ]] && [[ "$(stat -c %s "$MODEL/$name.part")" == "$expected_size" ]]; then
      break
    fi
    if [[ -f "$MODEL/$name.part" ]] && [[ "$(stat -c %s "$MODEL/$name.part")" == "$expected_size" ]]; then
      break
    fi
    if [[ "$attempt" == 5 ]]; then
      echo "Download failed after five attempts: $name" >&2
      exit 1
    fi
    echo "Resuming $name after interrupted transfer (attempt $attempt)" >&2
    sleep 5
  done
  if [[ "$(stat -c %s "$MODEL/$name.part")" != "$expected_size" ]]; then
    echo "Incomplete downloaded shard: $name" >&2
    exit 1
  fi
  mv "$MODEL/$name.part" "$MODEL/$name"
done
cd "$MODEL"
sha256sum config.json generation_config.json tokenizer_config.json tokenizer.json \
  vocab.json merges.txt model.safetensors.index.json model-*.safetensors > SHA256SUMS
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
tree = json.loads((root.parent / "source-tree.json").read_text(encoding="utf-8"))
checked = 0
for entry in tree:
    name = entry["path"]
    if not name.endswith(".safetensors"):
        continue
    expected = entry["lfs"]["oid"]
    size = entry["lfs"]["size"]
    path = root / name
    if path.stat().st_size != int(size):
        raise SystemExit(f"Size mismatch: {name}")
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if digest != expected:
        raise SystemExit(f"SHA256 mismatch: {name}")
    print(f"VERIFIED {name} {digest}")
    checked += 1
if checked != 4:
    raise SystemExit(f"Expected four weight files, verified {checked}")
PY
printf '{"repository":"Qwen/Qwen2.5-7B-Instruct","revision":"%s","metadata_transport":"Hugging Face immutable revision via mirror","weight_transport":"official Qwen ModelScope repository","weight_hashes_verified_against_huggingface_api":true,"hashes_file":"SHA256SUMS"}\n' "$REV" > SOURCE.json
echo BASE_DOWNLOAD_COMPLETE
