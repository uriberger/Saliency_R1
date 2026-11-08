# DeepEyes on Qwen3-VL: is there headroom left? — 2026-11-07

Prompt 1 of [deepeyes-build-prompts.md](deepeyes-build-prompts.md); the gate that
[baseline-port-estimates.md](baseline-port-estimates.md) §8 says to run before spending
3–5 weeks and 1,000–2,500 A100-hours on the arm.

The gate as specified: serve `ChenShawn/DeepEyes-7B`, run their `eval/eval_vstar.py` and
`eval/eval_hrbench.py`, and score it against Qwen3-VL-8B-Instruct and our cold start on
V*, HR-Bench and our `eval_mini` natural set.

**The harness is built, tested and committed. The three GPU arms did not run, because
this cluster has no GPU.** §4 is that blocker. §5–§7 are what can be decided without
them, which turns out to be most of it.

---

## Verdict

**NO-GO on the 80-step run as budgeted.** Three independent lines put the method's
expected gain on this backbone at **~+5 on V\***, not the paper's +18.9, and the
benchmarks that gain lives on are not in our suite.

1. **The vendor already measured it.** The Qwen3-VL tech report, §5.6: *"within the
   Qwen3-VL family, the absolute improvement by adding tools is consistently ~5 points
   across V\*."* That is DeepEyes' tool, measured by the people with the most incentive
   to make the number large, on the family we would train.
2. **The baseline has moved most of the way.** DeepEyes' +18.9 is measured from
   Qwen2.5-VL-7B at 71.2. Qwen3-VL-8B-Thinking is at 77.5 *with no tool at all*, and our
   own measurement puts Qwen3-VL-8B-**Instruct** at **74.0** on V*.
3. **On V\* the gain is not even RL-specific.** ZoomEye — training-free, a hand-written
   search workflow on the same 7B backbone — scores **90.6** on V* against DeepEyes'
   90.1, in DeepEyes' own Table 1.

And the part that decides it for *our* table: DeepEyes' entire advantage lives on V* and
HR-Bench, and `eval_mini` contains neither. On the two benchmarks that do overlap, their
own reported deltas are **POPE +1.8** and **MathVision +1.0** — inside the noise of a
100-row mini suite.

**For the writeup**, the sentence worth having either way:

> DeepEyes' claim that the ability "emerges from RL without cold-start SFT" cannot be
> tested on Qwen3-VL, because Qwen3-VL received exactly that cold start from its vendor:
> ~10k synthesized grounding traces, Qwen2.5-VL-32B SFT-ed into a "think → act → analyze
> feedback → answer" visual agent, ~120k distilled multi-turn agentic interactions, then
> cold-start SFT plus tool-integrated RL — and the official cookbook ships an
> `image_zoom_in_tool` whose schema is `bbox_2d` plus a label, the same tool.

---

## 1. The published table, read correctly

`baseline-port-estimates.md` §8 carries this table and it is right, but the V* column is
easy to misread, so here it is with the sub-columns that produce it. From DeepEyes'
Table 1 ("E2E indicates whether the model is end-to-end, requiring no manually defined
workflow; ∗ denotes results reproduced by ourselves"):

| model | E2E | size | V* Attr | V* Spatial | **V\*** | HR-4K | HR-8K |
|---|---|---|---|---|---|---|---|
| GPT-4o | ✓ | – | – | – | 66.0 | 59.0 | 55.5 |
| o3 | ✓ | – | – | – | **95.7** | – | – |
| SEAL | ✗ | 7B | 74.8 | 76.3 | 75.4 | – | – |
| DyFo | ✗ | 7B | 80.0 | 82.9 | 81.2 | – | – |
| **ZoomEye** (training-free) | ✗ | 7B | 93.9 | 85.5 | **90.6** | 69.6 | 69.3 |
| Qwen2.5-VL∗ | ✓ | 7B | 73.9 | 67.1 | **71.2** | 68.8 | 65.3 |
| Qwen2.5-VL∗ | ✓ | 32B | 87.8 | 88.1 | 87.9 | 73.9 | 70.4 |
| **DeepEyes** | ✓ | 7B | 91.3 | 88.2 | **90.1** | 75.1 | 72.6 |
| Δ vs Qwen2.5-VL-7B | | | +17.4 | +21.1 | **+18.9** | **+6.3** | **+7.3** |

Two readings that only appear once the sub-columns are visible:

* **71.2 is the V\* *overall*; 73.9 is its `direct_attributes` sub-column.** The overall
  is the row-weighted mean over the 191 questions (115 Attr + 76 Spatial):
  73.9·115/191 + 67.1·76/191 = 71.2. Their `judge_result.py` pools all rows for its
  "Overall Accuracy", so our harness computes the same quantity theirs does.
* **ZoomEye matches DeepEyes on V\* with no training.** DeepEyes' contribution over a
  hand-written zoom workflow is HR-Bench (+5.5 / +3.3) and being end-to-end — not the V*
  headline. The headline number is a property of *having a zoom loop at all*.

The rest of their paper is small. Grounding: refCOCO +0.7, refCOCO+ +1.0, refCOCOg +0.6.
POPE +1.8 overall, and **−1.9 on the Adversarial split**. Reasoning (Table 3): MathVista
+1.9, MathVerse +1.7, MathVision +1.0, WeMath +4.3, DynaMath +1.7, LogicVista +1.8.

## 2. Where our backbone starts

| model | V* | HR-4K | HR-8K | source |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 71.2 | 68.8 | 65.3 | DeepEyes Table 1 |
| Qwen3-VL-8B-Thinking, no tool | 77.5 | 72.4 | 68.1 | Qwen3-VL benchmark chart¹ |
| **Qwen3-VL-8B-Instruct, no tool** | **74.0** | – | – | **measured, this repo** |
| Qwen3-VL-235B-A22B **+ tools** | 93.7 | 85.3 | 82.3 | Qwen3-VL tech report §5.6 |
| DeepEyes-7B | 90.1 | 75.1 | 72.6 | DeepEyes Table 1 |

¹ The 77.5/72.4/68.1 row is **not** verifiable in the tech report's arXiv HTML — its
tables do not survive the LaTeX→HTML conversion. It traces to a rendering of the official
Qwen3-VL benchmark chart, and it is the **Thinking** variant. Treat it as secondary.

**The measured row is ours and it is solid**: `outputs/onetoken/onetoken-qwen3vl8b/`,
`Qwen/Qwen3-VL-8B-Instruct`, the `full` (unperturbed) arm, 100 of V*'s 191 questions,
greedy, at the model's own default resolution. Both independent scans of that probe agree
to the decimal — `onetoken` and `droptoken` each report 74.0% on the `full` arm — which is
a useful consistency check, since they were run as separate jobs. Caveats: it is the
Instruct model, not Thinking and not our cold start; it uses lmms-eval's MCQ prompt, not
DeepEyes' tool prompt; and n=100 gives a ±4.4 pp standard error, so it is "about the same
as Qwen2.5-VL-7B's 73.9 Attr sub-column", not a precise value.

That caveat cuts both ways and is worth stating plainly: **74.0 is close enough to
Qwen2.5-VL-7B's 71.2 that line 2 of the verdict is the weakest of the three.** The
vendor's own ~+5 (line 1) and ZoomEye (line 3) do not depend on it.

## 3. The in-repo HR-Bench numbers that look relevant and are not

`docs/fig1-benchmarks-numbers.md` and `outputs/fig1-multistep/bench_*.json` carry
per-chain grades on `hrbench4k_mini`, `hrbench8k_mini` and `vstar_mini` for our cold
start and for the overlap arm:

| scan | dataset | coldstart | ours |
|---|---|---|---|
| bench_b | hrbench4k_mini | 63.9% (n=97) | 68.5% (n=92) |
| bench_b | hrbench8k_mini | 60.2% (n=98) | 62.7% (n=83) |
| bench_d | vstar_mini | 60.2% (n=98) | 71.4% (n=91) |

**Do not put these in a table next to the numbers above.** `outputs/logs/fig1b-hrbench.runner.sh`
runs that scan with `--max-image-side 1024` and `--max-new-tokens 768`. HR-Bench-4K images
are 4,000 px and HR-Bench-8K's are 8,000; capping the long side at 1,024 removes precisely
the detail these two benchmarks exist to test, and the V* scan is capped the same way.
That is why the cold start reads 60.2 on `vstar_mini` while the base Instruct model reads
74.0 at full resolution — the gap is the resolution cap, not the cold start. The `n`
columns also differ by arm, because chains that produced no gradeable answer were dropped,
which biases toward whichever arm dropped more.

They are a valid *relative* ours-vs-coldstart signal under a fixed cap. They are not
benchmark numbers, and the gate still needs the real ones.

## 4. Why the three GPU arms did not run

This cluster has no GPU hardware, today, at all:

```fish
sinfo -h -N -o "%G" | grep -ci gpu     # -> 0
```

`batch_block1`, `batch_singlenode`, `batch_short`, `batch_long`, `batch_large`,
`backfill` and `interactive` all report `TotalNodes=0`. Cluster-wide there are **zero
running GPU jobs** — all 45 running jobs are on `cpu`, `cpu_long`, `cpu_short` or
`cpu_datamover` — and **32 pending at `Reason=PartitionConfig`**, which is what a
partition with no node behind it looks like. The partitions accept submissions; the jobs
never start.
This is [[gpu-partitions-can-be-empty-of-hardware]], still true a week on, and
`baseline-port-estimates.md` §9 records the same finding.

`cs-oci-ord-login-01` ([[second-a100-cluster]]) has the hardware and the same repo at the
same paths, but it is a separate login node reachable only over ssh, and ssh is outside
this session's permissions. So the gate's generate legs are staged, not run.

`launch_deepeyes_gate_job.sh` refuses to submit when that grep returns 0, rather than
queueing into a partition that cannot start it. `--force` overrides.

## 5. What *was* measured

**The judge leg, end to end, against the real gateway.** This is the one substitution the
build brief asks for — their self-hosted Qwen2.5-72B replaced by our hosted judge — and it
is pure text, so it needs no GPU. `test_deepeyes_gate_cpu.py --judge` drives both of their
graders over `azure/openai/gpt-4o-mini` at `https://inference-api.nvidia.com`: **17/17**,
including the fall-through cases where the rule-based shortcuts miss and the judge decides.
The substitution works and removes 4 GPUs from the job as the estimate claims.

That test also pins a leniency in *their* HR-Bench grader worth knowing before reading
their HR-Bench column: `judge_result_hrbench.py`'s third rule is `answer in pred_ans`,
where `answer` is a **one-character** gold letter. "the sign reads 37B" scores correct
against gold `B` on the substring, not on the content. Their V* grader does not have this
problem, because V*'s gold is always option A and it tests the string `"A. <gold text>"`.

**The eval_mini natural set**, already banked in `outputs/bench_baselines/`:

| arm | natural mean | mmerealworld | pope | realworldqa | mmstar |
|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | 0.648 | 0.310 | 0.882 | 0.720 | 0.681 |
| cold start (sft-coldstart) | 0.724 | 0.653 | 0.867 | 0.693 | 0.681 |
| overlap-8k (ours) | 0.747 | 0.697 | 0.890 | 0.700 | 0.700 |

The Instruct row is at n=100 against the others' n=300 and is **not comparable** in any
reasoning column — see [[plain-instruct-rows-are-not-comparable]]; its 0.310 on
mmerealworld is a 2-token-answer artefact, not a capability. It is here because the gate
asked for the natural set, not because it supports a comparison.

## 6. The decision, stated as a cost

The run is 30–71 h on 8×A100 for one 80-step pass, 4–9 chunked jobs, on top of ~3–5 weeks
of engineering (P0's `deepeyes_fork/` and `deepeyes` conda env already exist, built by a
sibling session). What it buys, on the best available estimate, is a baseline row that is
**~+5 on V\* and ~+2–3 on
HR-Bench** over a backbone we would have had anyway — on two benchmarks that are not in
`eval_mini`, against a method whose V* headline a training-free workflow already matches.

Two framings, because the answer differs and the brief only asks for one:

* **As "does the method still have headroom"** — the question §8 of the estimate poses —
  the answer is no. ~+5 is not +18.9, and the vendor measured it on our family.
* **As "we need a comparable row because a reviewer will ask about DeepEyes"** — headroom
  is not the criterion and the row is needed whatever its size. But then the cheap version
  is enough: cite `ChenShawn/DeepEyes-7B`'s published numbers and our measured backbone
  numbers side by side, which costs one 4-hour single-GPU job (§7), not a month.

**Recommendation: do not build the DeepEyes training arm. Run §7's 4-hour gate when GPUs
return, publish the measured comparison, and spend the month elsewhere.** If something in
§7 comes back far from expectation — see §8 — reopen it.

## 7. How to run it when GPUs return

Everything is staged. On a login node with internet (already done once; re-runs are no-ops):

```fish
# CPU, ~5 min. Patches their clone and converts HR-Bench parquet -> TSV (6.6 GB).
python build_deepeyes_eval_data.py --out-dir outputs/deepeyes_gate/data
```

Then, on a cluster with GPU nodes:

```fish
set -x NVIDIA_API_KEY (cat ~/.config/nvidia/api_key)
bash launch_deepeyes_gate_job.sh --name deepeyes-gate --duration 4
```

Three arms, serially, in one 1-GPU allocation: `deepeyes-7b` (`ChenShawn/DeepEyes-7B`),
`qwen3vl-8b` (`Qwen/Qwen3-VL-8B-Instruct`), `coldstart`. All three are **already on
disk** — DeepEyes-7B's 16.6 GB was pulled into `HF_HOME` for this gate, and the compute
nodes run `HF_HUB_OFFLINE=1`, so nothing downloads at run time.

`--bench vstar --duration 2` is the cheap version: 191 rows per arm, ~10 min each, and V*
is the benchmark the decision turns on. The full `vstar,hrbench` adds 1,600 HR-Bench rows
per arm, ~1 h per arm.

The baselines are scored **through DeepEyes' own zoom loop**, not through lmms-eval. That
matters: their claim is about what the loop buys, so a baseline measured without the loop
concedes the question before it is asked.

## 8. What would change the verdict

Pre-registered, so the result is read rather than argued:

| if the §7 run shows | then |
|---|---|
| DeepEyes-7B reproduces at 90.1 ± 2 **and** our cold start is below ~78 on V* | the ~12-point gap is real; reopen, but note ZoomEye gets it training-free |
| our cold start lands at or above ~85 on V* | settled, no-go, with a measurement instead of an inference |
| DeepEyes-7B fails to reproduce (say < 85) | their eval release cannot be run as published (§9) — treat *all* their numbers with care, including the baseline the +18.9 is measured from |
| our cold start is below ~70 on V* | the backbone is worse than Qwen2.5-VL-7B on this axis and the whole premise of §2 is wrong |

## 9. Three faults in their eval release

`deepeyes-build-prompts.md` says the eval scripts are "model-agnostic and need zero
changes". Model-agnostic, yes — the zoom tool is advertised as **plain text in the system
message**, not through the API's `tools=` field, so no chat template has to render a tool
call and a Qwen3-VL checkpoint is driven by the same code as a Qwen2.5-VL one. That half
is right, and it is why the harness is small.

"Zero changes" is wrong, three times, all in the HR-Bench leg:

| file | fault | when it bites |
|---|---|---|
| `eval_hrbench.py:321` | uses `pd.read_csv`, never imports pandas | `NameError` in `__main__`, before the first row |
| `eval_hrbench.py:149` | uses `copy.deepcopy`, never imports copy | `NameError` in the worker, reported as a dead pool |
| `judge_result_hrbench.py:39` | `args.vstar_bench_path` on a parser that only defines `--hrbench_path` | `AttributeError` at module level, before argparse prints help |

The third is the interesting one. It is a rename that was applied to the argument and not
to the one place it is read, and it fires before anything else can — which means **the
published HR-Bench legs cannot have been run from the published files**. That is not
evidence their numbers are wrong, but it is a reason to weight §8's third row.

A fourth thing is latent rather than fatal: both graders do `error_nums_type[test_type] += 1`
on the branch where the prediction contains the literal string `ERROR`, and `test_type` is
a `__main__` loop variable that does not exist in a `spawn`-started worker. It raises only
when a model's output contains "ERROR", which is rare but is exactly what a failed
generation can produce.

`build_deepeyes_eval_data.py` patches all three faults idempotently and asserts they took.
A `git pull` in the scratch clone silently restores all three;
`test_deepeyes_gate_cpu.py` is the regression guard.

## 10. Files

| path | what |
|---|---|
| `build_deepeyes_eval_data.py` | stages + patches their clone, parquet → TSV, CPU-only |
| `launch_deepeyes_gate.sh` | `_run.sh`: serve on vLLM, generate, judge, bank |
| `launch_deepeyes_gate_job.sh` | `_job.sh`: three arms in one allocation; refuses a GPU-less cluster |
| `test_deepeyes_gate_cpu.py` | 17 checks; `--judge` adds the live gateway leg |
| `outputs/deepeyes_gate/data/` | `hr_bench_{4k,8k}.tsv` (1.9 + 4.6 GB) + `manifest.json` |
| `/home/uberger/scratch/deepeyes_ref/DeepEyes` | their clone — **reference only**, never executed for training |

Staged into the shared `HF_HOME`: `ChenShawn/DeepEyes-7B` (16.6 GB),
`Qwen/Qwen3-VL-8B-Instruct`, `craigwu/vstar_bench` (191 rows, verified),
`DreamMr/HR-Bench` (800 + 800).

## Sources

* [arXiv:2505.14362](https://arxiv.org/abs/2505.14362) — DeepEyes (ICLR 2026); Tables 1–3
* [Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes) — their code, reference only
* [ChenShawn/DeepEyes-7B](https://huggingface.co/ChenShawn/DeepEyes-7B) — the checkpoint
* [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) — §5.6, the ~5-point tool gain
* [baseline-port-estimates.md](baseline-port-estimates.md) §8–9 — the estimate this executes
* [deepeyes-build-prompts.md](deepeyes-build-prompts.md) — Prompt 1 is this document
