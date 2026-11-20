# Tongji HPC Qwen Evidence Training

This directory contains the isolated HPC workflow for the Qwen evidence experiments.
It uses the supplied Tongji HPC skill for SSH/SFTP and Slurm; credentials are
read only from the supplied credential document into a process environment
variable and are never included in the training package.

The remote root is `/share/home/u24121/ecocompanion_qwen_v1`. The preparation
job reuses the existing CUDA/PyTorch stack in an isolated virtual environment
and only adds `bitsandbytes`. The immutable Hugging Face revision supplies the
official weight hashes; the official Qwen ModelScope repository supplies the
weights, which are verified before use. `download_qwen_base.sh` never appends
to existing weights and promotes only full-size temporary downloads.
The L40 training job then runs QLoRA, performs a development-only adapter
smoke evaluation, and creates a package with hashes and provenance.

Only project-authored synthetic examples are used. Local conversations,
reports, API keys, QQ/WeChat configuration and Windows credentials are excluded.
The 9-row held-out test file was used only after training for the frozen
comparison; its SHA-256 matches the local frozen copy. Training a small
synthetic fixture is not clinical validation.

Scripts are numbered by the Slurm workflow itself:

1. `download_base.sbatch` on `intel`, then `validate_base.sbatch` on `intel`.
2. `train.sbatch` on one `L40` GPU, submitted after base validation succeeds.
3. `train_v2.sbatch` for the independent 10-epoch diagnostic experiment.
4. `heldout_v2.sbatch` for the fixed base-versus-v2 Transformers comparison.

No existing remote job, model or environment is overwritten.

## Results, 2026-10-29

- Base `3180384`: all four shards passed the pinned Hugging Face SHA-256 values.
- Base `3180385`: complete BF16 load passed, 7,615,616,512 parameters.
- v1 `3181073`: 2 epochs, 20.42 s training, validation loss 1.3373;
  0/3 valid development JSON outputs. Package retained, not deployed.
- v2 `3181155`: 10 epochs, 114.6 s training, validation loss 0.3894;
  3/3 valid development JSON outputs, but one invented evidence quote.
- Held-out `3181768`: v2 achieved 9/9 valid JSON and 9/9 safety-signal matches;
  8/9 evidence grounding. `quotation-001` paraphrased evidence rather than
  quoting it exactly. Unconstrained base decoding yielded 0/9 valid JSON.
  Ollama constrained base decoding yielded 9/9 valid, 9/9 grounded, and 4/9
  safety-signal matches; it is a different inference condition, not a claim of
  equivalent quantization or a direct controlled comparison.

Neither adapter is deployed. The current `qwen2.5:7b` application model is
unchanged. `backend/tests/test_pet.py` passed all 36 tests after the work.
The nine-row fixture cannot establish clinical performance or support a
psychological diagnosis. Future tuning requires new training/development data;
do not continue tuning against this already inspected test set.

The v2 archive originally embedded the v1 development result because the
packaging script used a fixed v1 filename. All seven listed adapter-file hashes
and the packaged v2 adapter SHA-256 were independently verified. The archive
manifest now records the provenance error and the actual development result;
the adapter weights and training data are unchanged. `package_adapter.py` now
requires an explicit `--validation` path to prevent this mismatch.

## v3 reviewed external-data experiment, 2026-10-29

The author-released `qiuhuachuan/PsyDial-D4` at revision
`0ceef5ac19fb8174baa6be31ce1b2637711cb4c2` is now used selectively.
Its data card declares Apache-2.0, Chinese, 2,382 reconstructed long-term
support dialogues. The downloaded source is 35,544,509 bytes with SHA-256
`29bfa723645f3c7213d00e34042c164545b27629487253fa6ebec97c7f3638b9`.
Thirty-nine seeker turns were individually reviewed for conservative emotion,
stress and support-need labels. The full dialogues were not used for training.
No assistant reply, diagnosis, or safety keyword was converted into a label.

The new frozen training configuration contains 100 rows: 83 train and 17
development, with zero new test rows. It includes the previous 29 boundary
examples and 32 new independent boundary groups. Every evidence quote was
checked against the current message. External dialogues cannot cross the
train/development split. The preparation script does not read the previous
frozen test file, whose SHA-256 remains
`8bb987af81de8c4d6f758ffca58d5f120ea96dd29415b5ca6350ceb1a1688e08`.

The raw corpus is excluded from source handoff; exact dialogue/turn provenance,
the ACL 2025 citation and license URL are retained in `training/data/v3/`.
The application and Ollama model remain unchanged during this experiment.

### Actual v3/v4 outcomes

- v3 `3183901` (A800): 100 rows, 83 train/17 development, 6 epochs,
  226.5 seconds. Development: 17/17 valid JSON, 16/17 grounded, 16/17 safety
  matches. One quote was lightly paraphrased and ordinary panic became an
  uncertain safety signal. `3183904` correctly refused the frozen regression.
- v4 `3184722` (A800): one development-only correction, with 12 new external
  dialogues and 8 new boundary groups. 120 rows, 97 train/23 development,
  6 epochs, 264.5 seconds. Development: 23/23 valid JSON, 23/23 grounded and
  23/23 safety matches.
- Frozen regression `3184723`: 9/9 valid JSON, 9/9 safety matches, 8/9
  grounded. `current-001` produced `我已经准备好了药`, adding `我` to the
  original `已经准备好了药`. The deployment gate remains failed.

Both v3 and v4 archives were downloaded and all seven listed adapter-file
hashes passed. They remain research artifacts. No adapter is merged, imported
into Ollama, or activated. The inspected nine-row fixture must not be used for
further tuning. Future work needs independently authored training/development
data, especially exact-span extraction under context compression, followed by
new held-out tests and a quantized Ollama regression.
