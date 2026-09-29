# Handoff prompt: select the Omni's saliency head pair

> **DONE, 2026-09-29 — `docs/omni-head-selection.md` is the result.** The pick is
> `--overlap-layer 19 --overlap-heads 4,9`, the incumbent `33 / 28,31` ranks 148–176 of
> 192, and the probes now go through the family seam with `test_probe_family_cpu.py`
> holding Qwen3-VL still.
>
> **Read §4 of that doc before running the training arm.** The correctness label this
> selection is built on is 52% disputed — the Omni is a base checkpoint and
> `accuracy_reward`'s exact-string fallback cannot read the prose it writes — and the
> honest next step is a cold-started Omni rather than an arm on `19 / 4,9`.
>
> Everything below is the original prompt, kept because §"The work" is still the procedure.

Run the head-selection procedure on `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`,
so the overlap reward stops being defined on two arbitrary heads.

## Read these first, in this order

| file | what it gives you |
|---|---|
| `docs/omni-training-harness.md` | the whole Omni harness: two conda envs, the second TRL clone, the geometry seam, and §9's list of the nine things that broke |
| `head_correlation_probe.py` (the docstring) | the procedure itself, and why the parity split is the part that matters |
| `launch_head_correlation.sh` | how it shards across a node |
| `intervene_probe.py --stage prepare` | what builds the cases the scan reads |
| `vlm_family.py` → `NemotronVL` | the Omni's geometry, already solved on the measuring side AND on the trainer side |
| `nemotron_loader.py` | the eight shims this checkpoint needs to load at all |

Memory entries `omni-grpo-harness`, `omni-grpo-two-silent-traps`,
`omni-completion-cap-is-the-ceiling` and `omni-grpo-step-time` cover the same ground short.

## Why this is the next thing

The Omni GRPO run is built, verified and has 30 steps on the board
(`outputs/omni_grpo_plan_a/grpo-omni30b-overlap__wov0.2_L33_h28-31_mean_in_c768/`). What it
trained is `--overlap-layer 33 --overlap-heads 28,31`, and **neither number was selected**:

* **33** is the attention layer nearest Qwen3-VL's layer 22 in relative depth (63% against
  61%). That is the entire argument for it.
* **28,31** were chosen on Qwen3-VL-8B by `head_correlation_probe.py`. On any other model
  the same two indices name two arbitrary heads.

So those 30 steps are a harness smoke test and nothing else — the reward moved, which says
the pipeline works, not that it rewarded anything meaningful. **The real arm has to start
from selected heads**, because the layer and the head pair ARE the reward's definition.
Do not treat the 30-step checkpoint as a baseline to continue from.

## What makes this cheaper here than it was on Qwen3-VL

**The search is 192 cells, not 1,152.** Only 6 of the Omni's 52 decoder layers have an
attention matrix at all — `hybrid_override_pattern` puts them at `[5, 12, 19, 26, 33, 42]`
— so 6 x 32 heads is the whole space, and the parity split has far less multiplicity to
survive. It also dissolves the layer question: the scan covers all six, so "33 by depth"
stops being an assumption rather than being defended.

## The work

### 1. Build the cases

`head_correlation_probe.py` does not generate anything — it reads the chains and the
per-step Grounding-DINO unions an `intervene_probe --stage prepare` run already wrote, and
`--cases-dir` points at that directory. So that pass has to run on the Omni first.

Check whether `intervene_probe.py`'s `prepare` stage is model-agnostic before assuming it
is; it predates the family seam. If it hardcodes Qwen3-VL anywhere, give it the same
treatment as §2.

### 2. Port the probe through the family seam

`head_correlation_probe.py` is Qwen3-VL-only in exactly two places, and both have an
established fix in this repo:

* **line ~118**: `if type(m).__name__ == "Qwen3VLTextAttention"`, raising
  `RuntimeError("no Qwen3VLTextAttention modules found")` below it. Ask the family:
  `vlm_family.family_for(model).attn_classes`, and use `attention_layers(model)` for which
  layer indices exist — on a hybrid most of them do not, and "the scan saw fewer layers
  than the model has" is the CORRECT outcome here and a bug anywhere else.
* **line ~267**: `PROBE.load_model(...)`, which resolves the architecture as
  `getattr(transformers, config.architectures[0])` and raises AttributeError on a
  `trust_remote_code` checkpoint. Use `nemotron_loader.load_model` — it is the same set of
  shims `sink_location_probe` and the trainer already share, so do not write a third copy.

Two more things the trainer needed and this probe will too:

* the image token is **18** (`config.img_context_token_id`), not 151655. The family
  resolves it; `Family.image_token_id` after `bind`.
* the patch grid comes from `imgs_sizes` over a 32px token, a different size per picture —
  `NemotronVL.token_grid` / `grids_for`. There is no `image_grid_thw`.

`test_vlm_geometry_cpu.py` is the pattern for proving the port did not move Qwen3-VL: state
each replaced expression and check the family's answer against it.

### 3. Scan and report

```
bash launch_head_correlation.sh --gpus 8 --out-dir <dir> --cases-dir <prepare-dir> \
     --base-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
python head_correlation_probe.py --stage report --out-dir <dir>
```

**Fix `--max-union` before you look at the confirmation half, not after.** The probe's own
docstring says why: every map measured so far reads lower the larger the union gets
(r(union, auroc) = -0.55 averaged over all 1,152 Qwen3-VL heads), the median step's union
covers 54% of the grid, and a threshold chosen after seeing the result is a researcher
degree of freedom. `report` prints the level by union decile first for exactly this reason.

**Read the parity split, not the ranking.** Heads are ranked on odd-indexed rows and
re-scored on even ones. A head that survives is a candidate; one that does not is selection
noise, and on 192 cells that is a much stronger statement than it was on 1,152.

### 4. Then, and only then, the training arm

Re-run `launch_grpo_omni_overlap_job.sh` with the selected `--overlap-layer` and
`--overlap-heads`. Everything else about the launcher is settled and documented.

## Things that will cost you a day if you rediscover them

* **Two conda envs.** `nemotron` (torch 2.8, transformers 5.13.0.dev0) for anything that
  touches the model; `nemotron_vllm` (torch 2.11, vLLM 0.20.2) only for the generation
  server. The probe belongs in `nemotron`.
* **`import trl` resolves through a PEP 660 meta-path finder**, so `PYTHONPATH` cannot
  redirect it. Both envs have `pip install -e trl_repo_nemotron`. If you change anything
  under `trl/`, run `bash patch_trl_nemotron.sh` — and note it patches a SHARED clone.
* **`export SR1_REPO=<repo>`** on compute nodes, so a copy of `nemotron_loader.py` living
  inside `trl_repo_nemotron` can still find `vendor/mamba_ssm_min`. Without that vendored
  layernorm-only `mamba_ssm`, the Nemotron decoder raises at *import*.
* **`export HF_HOME=/home/uberger/scratch/cache/hf_cache`, `HF_HUB_OFFLINE=1`.** Compute
  nodes have no internet; pre-fetch from the login node.
* **A per-rank Grounding-DINO costs ~8 GB a card.** If the prepare stage grounds anything,
  point it at a DINO server rather than letting every rank build its own — that one cost
  peak 74.1 GB against 66.4 and 123 retry lines in half an hour, with the detector's own
  card idle. See `omni-grpo-two-silent-traps`.
* **The Omni's chat template opens `<think>` in the prompt.** Completions carry only the
  closing tag. Anything that parses the format has to know, or it scores every rollout
  malformed and reports `format 0.000` on a model that is fine.
* **ptrace is off on these nodes**, so py-spy and gdb cannot attach. `faulthandler` +
  `dump_traceback_later` from inside the process is the way in —
  `omni_vllm_probe.py --watchdog` is the worked example.
* **Read CLAUDE.md**: the launch directory is read-only, all changes happen in a worktree
  via `./worktree.sh new <branch>`, and the user works in fish.
* **Do not score test benchmarks during training**; dispatch `watch_bench_evals.sh` after.

## Done means

A reported head pair (and layer) on the Omni that survives the odd/even parity split at a
`--max-union` threshold fixed in advance, the probe ported through the family seam rather
than forked, and a CPU test that shows Qwen3-VL's answers are unchanged by the port.
