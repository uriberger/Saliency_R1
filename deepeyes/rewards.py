#!/usr/bin/env python
"""DeepEyes' reward composite, on our hosted judge and on verl's `num_turns`.

A TRANSCRIPTION of `verl/utils/reward_score/vl_agent.py` from Visual-Agent/DeepEyes
(cloned to scratch, reference only), with two substitutions that are stated rather than
hidden. `docs/deepeyes-build-prompts.md` Prompt 5 is the brief.

    0.8*R_acc + 0.2*R_format + 1.2*R_tool          vstar / vl_agent / chart
    0.8*R_acc + 0.2*R_format + 1.2*R_tool          geoguessr (different answer parse)
    1.2*R_acc + 0.4*R_format                       thinklite_eureka / xince (no tool)

    R_tool   = 1 iff (at least one zoom happened AND R_acc > 0.5)
    R_format = -1 on any tag MISCOUNT, else 0
    R_acc    = binary

THE CONDITIONAL GATE IS THE DESIGN. The tool bonus is 1.5x the accuracy weight and it
only pays out on a correct answer, so zooming indiscriminately buys nothing. It is
preserved exactly: `tool = 1.0 if zoomed and acc > 0.5 else 0.0`.

---------------------------------------------------------------------------
WHAT "ANY TAG MISMATCH" ACTUALLY MEANS -- read this before trusting R_format
---------------------------------------------------------------------------

Their predicate compares COUNTS. It is not "the tags must be present":

    count("<think>") != count("</think>")                              -> error
    count("<|vision_start|><|image_pad|>") != count("<|image_pad|><|vision_end|>")
    count("<answer>") != count("</answer>")   (AFTER the last </think>) -> error
    len(answer_text) >= 1000                                           -> error

A completion with NO tags at all scores 0 == 0 on every line and is NOT penalised. This
matters, because `docs/deepeyes-build-prompts.md` Prompt 4 reads R_format as "a model
that cannot hit the tag structure is actively penalised on every rollout". Under their
actual predicate it is not: a tagless rollout gets R_format = 0, and only an UNBALANCED
one gets -1. The cold-start argument survives on R_tool and on the answer parse (below),
not on the format penalty. Measured, not assumed -- `test_deepeyes_rewards_cpu.py` pins
the tagless case at exactly 0.0.

The answer parse is where a tagless model really does lose. `compute_score` takes

    predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

so with no tags the WHOLE completion -- chain of thought included -- is handed to the
judge, and the >= 1000-character hack guard then fires on any long chain and turns
R_acc to 0 AND R_format to -1. That is the real cold-start argument.

---------------------------------------------------------------------------
SUBSTITUTION 1 -- the judge
---------------------------------------------------------------------------

They self-host Qwen2.5-72B on 4 GPUs and ask it for a literal "1" or "0". We use the
NVIDIA gateway judge that `llm_judge.py` / `trl/rewards/openai_rewards.py` already
carry, which returns 1-5 mapped to [0, 1]. R_acc is binary in their composite, so ours
is binarised at `> JUDGE_BINARISE_AT` (0.5) -- i.e. the judge must say 4 or 5 out of 5.
The same cut is the gate `acc > 0.5` uses, so the two cannot drift apart. A different
grader is a caption-level deviation; say so in the writeup.

FAILURE SEMANTICS, decided deliberately (Prompt 5 asks for this explicitly):

  tier 1  transient gateway errors are retried 5x with exponential backoff, which is
          `llm_judge.ask`'s rule and covers the common case.
  tier 2  a judge that still will not answer -- timeout, refusal, content filter,
          unparseable verdict -- falls back to the OFFLINE grader in
          `answer_grading.grade` (strict + soft, MCQ-aware, no network). This is the
          repo's "a judge refusal has no letter" convention applied the right way
          round: a refusal degrades grading quality, it does not fabricate "wrong".
  tier 3  only if the offline grader is also undecided -- `kind == "none"`, which
          happens when the row has no ground truth at all -- is R_acc 0.0. It is
          flagged `judge_failed=1`, `acc_source=ACC_UNKNOWN`, and printed. R_tool is
          then 0 by construction, which is the honest reading: we cannot certify the
          answer, so we do not pay the tool bonus for it.
  empty   a completion with no answer text is 0.0 with no call at all, matching
          `llm_judge.judge_scores`. That is wrong, not unmeasurable.

NEVER NaN. `llm_judge` and `openai_rewards` both mask a failure to None, and that is
right for an ANALYSIS script. It is wrong here: verl writes this number straight into
`reward_tensor` and GRPO's advantage is a group mean/std over it, so one NaN silently
destroys all 16 rollouts of that prompt instead of one. 0.5 ("neutral") was also
rejected -- it injects a number nobody measured into the group baseline.

Every sample reports `judge_failed` and `acc_source` in `reward_extra_info`, so the
rate is a logged metric rather than a thing to find out about later.

---------------------------------------------------------------------------
SUBSTITUTION 2 -- "a zoom happened"
---------------------------------------------------------------------------

`vl_agent.py:199` counts `<|vision_start|><|image_pad|>` in the response STRING. That
trick transfers in principle -- Qwen3-VL has identical vision token IDs (vision_start
151652, vision_end 151653, image_token 151655) -- but it is DEAD in this fork, and
silently:

    DeepEyes   verl/workers/reward_manager/naive.py:71
               response_str = self.tokenizer.decode(valid_response_ids)
    this fork  verl/workers/reward_manager/naive.py:79
               response_str = self.tokenizer.decode(valid_response_ids,
                                                    skip_special_tokens=True)

`<|vision_start|>`, `<|image_pad|>` and `<|vision_end|>` are special=True in the cold
start's tokenizer.json, so they are STRIPPED. Both counts are 0, they are equal, the
format check never fires, and `count_vision_1 > 0` is False on every rollout -- R_tool
would be identically 0 and the arm would be plain GRPO with a format penalty. Nothing
would have reported it.

So the zoom is re-derived from verl's `num_turns`, which `tool_agent_loop.py:180`
emits as `user_turns + assistant_turns + 1`: one generate round is 2, and each tool
round adds one user turn (the observation splice) and one assistant turn, so

    n_tool_rounds = (num_turns - 1) // 2          zoomed = n_tool_rounds > 0

`<think>` / `</think>` (ids 151667/151668) are added tokens with special=False and
`<answer>` / `</answer>` are not tokens at all, so those three count checks DO survive
the skip -- only the vision one dies. The vision check is still implemented, for
fidelity and because it is live when the caller hands us a non-skipped decode, and
`vision_miscount` is logged so "it never fires" stays a measurement.

If `num_turns` is missing the module does NOT quietly report "no zoom". It falls back
to counting `<tool_call>` (added token, special=False, survives the skip), sets
`turns_missing=1` and warns once. The fallback is the thing a model can fake by typing
the literal text, which is exactly why it is the fallback and not the rule.

---------------------------------------------------------------------------
HOW TO WIRE IT
---------------------------------------------------------------------------

verl 0.7.0.dev0 loads a custom reward by FILE PATH (`verl/trainer/ppo/reward.py:61`,
`get_custom_reward_fn`), and it does so BEFORE it resolves the reward manager name
(same file, :136 then :148). So this one file provides both halves and the fork needs
no edit at all:

    custom_reward_function.path=$REPO/deepeyes/rewards.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=deepeyes \
    data.reward_fn_key=data_source

`reward_manager=deepeyes` is the batched manager defined at the bottom of this file.
Use it. The stock `naive` manager calls `compute_score` once per sample SERIALLY, and
at 256 prompts x 16 rollouts that is 4,096 judge calls in a row -- an hour per step on
a 1 s judge. The batched manager thread-pools them (`JUDGE_MAX_WORKERS`, default 16)
and is otherwise identical. `naive` still works and is still correct; it is just slow.

THE ENV VARS HAVE TO REACH THE RAY ACTOR. The manager is constructed inside
`main_ppo.TaskRunner`, so NVIDIA_API_KEY / OPENAI_BASE_URL / JUDGE_MODEL must be in
that actor's environment -- i.e. in `ray.init(runtime_env={"env_vars": ...})`, not
merely in the submitting shell. Prompt 2 found that `RLHFDataset.__init__` can trigger
an implicit `ray.init()` with no runtime_env if the dataset is built first; if that
happens the judge will 401 on every sample and tier 2 will quietly carry the whole run
on the offline grader. `judge_failed` and `acc_source` in the step metrics are how you
see it. Do not set `reward_model.launch_reward_fn_async=True` without checking the
same thing for `compute_reward_async`'s worker.

    python test_deepeyes_rewards_cpu.py        # CPU, no GPU, no gateway (judge mocked)
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# This module is loaded by verl through importlib.util.spec_from_file_location, which
# does NOT put its directory (or the repo root) on sys.path. `llm_judge`,
# `answer_grading` and the gateway settings live at the repo root, one level up.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import answer_grading as AG  # noqa: E402

# ---------------------------------------------------------------------------
# the constants, read off vl_agent.py
# ---------------------------------------------------------------------------
#: `return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward` -- "reward 2" in
#: their file, the one that is not commented out.
W_ACC = 0.8
W_FORMAT = 0.2
W_TOOL = 1.2
#: `compute_score_math`: `return 1.2 * acc_reward + 0.4 * format_reward`. No tool term.
MATH_W_ACC = 1.2
MATH_W_FORMAT = 0.4
#: "Penalize for model trying to predict longer answer to hack llm-as-judge".
ANSWER_CHAR_LIMIT = 1000
#: their R_format is -1, not 0. The magnitude is the whole point of the tag grammar.
FORMAT_PENALTY = -1.0
#: our judge is 1-5 -> [0,1]; theirs is a literal 1/0. 4-or-5 is correct. Deliberately
#: the same number the composite's own `acc > 0.5` gate uses.
JUDGE_BINARISE_AT = 0.5

#: `verl/utils/reward_score/__init__.py:64-74` in their repo -- the routing table.
VL_DATA_SOURCES = frozenset({"vstar", "vl_agent", "chart"})
REASONING_DATA_SOURCES = frozenset({"geoguessr"})
MATH_DATA_SOURCES = frozenset({"thinklite_eureka", "xince"})

ROUTE_VL = "vl"
ROUTE_REASONING = "reasoning"
ROUTE_MATH = "math"

#: `acc_source` codes. Numeric because `reward_extra_info` is turned into an np.array
#: and fed to `process_validation_metrics`, which takes means of it.
ACC_JUDGE = 0        # the gateway judge answered
ACC_RULE = 1         # math_verify agreed without asking anyone
ACC_FALLBACK = 2     # tier 2: judge would not answer, offline grader decided
ACC_EMPTY = 3        # no answer text at all -> 0.0 with no call
ACC_UNKNOWN = 4      # tier 3: nothing could decide. flagged, counted, scored 0.0

_BOXED_THEIRS = re.compile(r"\\boxed\{([^}]+)\}", re.DOTALL)
#: balanced-brace version, from `deepeyes_fork/verl/utils/reward_score/acc_reward.py`.
#: NOT used for the reward -- their regex is. It exists so `boxed_nested` can count how
#: often `[^}]+` truncates a nested expression, which is the LogicVista/MathVision
#: parser-artefact shape and is worth knowing before it is argued about.
_BOXED_BALANCED = re.compile(
    r"(?:\\{1,2}boxed|\x08oxed)\{"
    r"((?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*)"
    r"\}",
    re.DOTALL,
)

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    HAVE_MATH_VERIFY = True
except Exception:  # pragma: no cover - depends on the env, not on the logic
    _mv_parse = _mv_verify = None
    HAVE_MATH_VERIFY = False


# ---------------------------------------------------------------------------
# the format predicate -- counts, not presence. See the module docstring.
# ---------------------------------------------------------------------------
def tag_flags(solution_str: str) -> dict:
    """The individual clauses of `is_format_error`, so each can be logged and tested.

    Returns `think_miscount` / `vision_miscount` / `answer_miscount` as 0.0/1.0 floats.
    `vision_miscount` is structurally unreachable when the caller decodes with
    `skip_special_tokens=True` (both counts are 0); it is kept, and counted, so that
    "it never fires" is a measurement instead of an assumption.
    """
    s = solution_str or ""
    think = float(s.count("<think>") != s.count("</think>"))
    vision = float(s.count("<|vision_start|><|image_pad|>") != s.count("<|image_pad|><|vision_end|>"))
    # their `predict_no_think`: everything after the LAST </think>. A chain that never
    # closed its think block leaves this as the whole string, which is their behaviour.
    tail = s.split("</think>")[-1].strip()
    answer = float(tail.count("<answer>") != tail.count("</answer>"))
    return {"think_miscount": think, "vision_miscount": vision, "answer_miscount": answer}


def extract_answer_vl(solution_str: str) -> str:
    """`compute_score`'s parse: `split("<answer>")[-1].split("</answer>")[0].strip()`.

    Note what this does with NO tags: it returns the whole completion. That is their
    behaviour and it is load-bearing -- it is what makes the 1000-character hack guard
    fire on a tagless chain of thought. Do not "fix" it into returning "".
    """
    return (solution_str or "").split("<answer>")[-1].split("</answer>")[0].strip()


def extract_answer_tagged(solution_str: str) -> Optional[str]:
    """`compute_common_reasoning`'s parse: a real `<answer>...</answer>` or None.

    Theirs runs it on `predict_no_think` (after the last `</think>`) via
    `extract_answer`, and a None is itself a format error there.
    """
    tail = (solution_str or "").split("</think>")[-1].strip()
    m = re.search(r"<answer>(.*?)</answer>", tail, re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# "a zoom happened", from verl's num_turns
# ---------------------------------------------------------------------------
_warned_turns_missing = [False]


def tool_rounds_from_turns(num_turns) -> int:
    """`num_turns` -> how many tool rounds the rollout actually ran.

    `tool_agent_loop.py:180` emits `user_turns + assistant_turns + 1`. A no-tool
    rollout is 1 assistant turn + 0 user turns + 1 = 2. Each tool round splices one
    observation (user_turns += 1, :379) and generates again (assistant_turns += 1,
    :229), so num_turns = 2 + 2k. The odd case -- a tool response that hit
    max_assistant_turns before generating again -- is covered by the ceil form.
    """
    if num_turns is None:
        return -1
    try:
        n = int(num_turns)
    except (TypeError, ValueError):
        return -1
    return max(0, (n - 1) // 2)


def zoom_from(num_turns, solution_str: str) -> tuple[bool, int, float]:
    """-> (zoomed, n_tool_rounds, turns_missing).

    num_turns is the rule. The `<tool_call>` count is the fallback and is announced:
    it is a token the model can type, which is the failure mode the num_turns route
    exists to avoid.
    """
    k = tool_rounds_from_turns(num_turns)
    if k >= 0:
        return k > 0, k, 0.0
    k = (solution_str or "").count("<tool_call>")
    if not _warned_turns_missing[0]:
        _warned_turns_missing[0] = True
        print(
            "[deepeyes-reward] WARNING: num_turns is absent from extra_info. R_tool is "
            "falling back to counting '<tool_call>' in the response text, which a model "
            "can fake. Use reward_manager=deepeyes (this file) or the fork's naive "
            "manager, both of which pass __num_turns__ through. turns_missing=1 on "
            "every sample until this is fixed.",
            flush=True,
        )
    return k > 0, k, 1.0


# ---------------------------------------------------------------------------
# the judge: tier 1 (gateway, retried) and tier 2 (offline grader)
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are an intelligent chatbot designed for evaluating the correctness of "
    "generative outputs for question-answer pairs. Your task is to compare the "
    "predicted answer with the correct answer and determine if they match "
    "meaningfully. Here's how you can accomplish the task:"
    "------"
    "##INSTRUCTIONS: "
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)

_client_lock = threading.Lock()
_client = [None]


def _judge_client():
    """Lazily built, and never raises at import -- `openai_rewards.py`'s convention.

    This module is imported by verl whether or not a judge call is ever made, and a
    missing key must surface as a counted tier-3 failure rather than as an exception
    inside the first training step.
    """
    with _client_lock:
        if _client[0] is None:
            import openai

            key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not key:
                print(
                    "[deepeyes-reward] WARNING: neither NVIDIA_API_KEY nor "
                    "OPENAI_API_KEY is set in this process. Every judged sample will "
                    "fall through to the offline grader (acc_source=ACC_FALLBACK). If "
                    "this is a training run, the env vars did not reach the ray actor.",
                    flush=True,
                )
                key = "xxx"
            _client[0] = openai.OpenAI(
                api_key=key,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://inference-api.nvidia.com"),
            )
        return _client[0]


def judge_binary(question: str, gold: str, answer: str, attempts: int = 5) -> Optional[float]:
    """-> 1.0 / 0.0, or None if the gateway would not answer. Tier 1.

    The prompt is `llm_judge.py`'s verbatim, so a judgement cached or reported by any
    other script in this repo is the same object. The 1-5 score is binarised at
    JUDGE_BINARISE_AT.
    """
    client = _judge_client()
    model = os.environ.get("JUDGE_MODEL", "azure/openai/gpt-4o-mini")
    content = (
        f"I will give you an image and the following text as inputs:\n\n"
        f"1. **Question Related to the Image**: {question}\n"
        f"2. **Ground Truth Answer**: {gold}\n"
        f"3. **Model Predicted Answer**: {answer}\n\n"
        "Your task is to evaluate the model's predicted answer against the ground "
        "truth answer, based on the context provided by the image and the question. "
        "Consider the following criteria for evaluation:"
        "- **Relevance**: Does the predicted answer directly address the question "
        "posed, considering the information provided in the image?"
        "- **Accuracy**: Compare the predicted answer to the ground truth answer. "
        "Does the prediction accurately reflect the information given in the ground "
        "truth answer without introducing factual inaccuracies?"
        "**Output Format**:"
        "Score: <a integer score of quality from 1-5>")
    for attempt in range(attempts):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=512, timeout=120,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": content}])
            m = re.search(r"Score:\s*(\d+)", r.choices[0].message.content or "")
            if not m:
                return None          # a refusal has no number; tier 2 decides
            return 1.0 if (int(m.group(1)) - 1.0) / 4.0 > JUDGE_BINARISE_AT else 0.0
        except Exception as e:
            if attempt == attempts - 1:
                print(f"[deepeyes-reward] judge gave up on one item: "
                      f"{type(e).__name__}: {str(e)[:140]}", flush=True)
                return None
            time.sleep(0.2 * 2 ** attempt)
    return None


def offline_grade(answer: str, gold: str) -> Optional[float]:
    """Tier 2. `answer_grading.grade`, binary, soft-or-strict. -> 1.0/0.0 or None.

    `grade` returns `{"strict": None, "soft": None, "kind": "none"}` when there is no
    gold string at all, which is the one case nothing offline can decide. Everything
    else -- MCQ letter, exact match, substring -- it answers without a network call.
    The SOFT grade is taken because this module's `answer` is the model's own prose,
    and the strict rule marks a correct verbose answer wrong (the artefact
    `answer_grading`'s docstring exists for).
    """
    g = AG.grade(answer or "", gold or "")
    if g["kind"] == "none" or g["soft"] is None:
        return None
    return 1.0 if g["soft"] else 0.0


def rule_math_verify(ground_truth, model_answer) -> bool:
    """`vl_agent.rule_math_verify`, with the hang guard the repo already uses.

    SIGALRM only works on the main thread, so it is armed only there; in a worker
    thread the call is simply wrapped. math_verify raising is not a reason to kill a
    training step -- it is a reason to ask the generative verifier.
    """
    if not HAVE_MATH_VERIFY:
        return False
    armed = False
    try:
        if threading.current_thread() is threading.main_thread():
            import signal

            def _boom(signum, frame):
                raise TimeoutError("math_verify timed out")

            signal.signal(signal.SIGALRM, _boom)
            signal.alarm(5)
            armed = True
        return bool(_mv_verify(_mv_parse(ground_truth), _mv_parse(model_answer)))
    except Exception:
        return False
    finally:
        if armed:
            import signal

            signal.alarm(0)


# ---------------------------------------------------------------------------
# prepare / finalize -- one implementation, two call shapes
# ---------------------------------------------------------------------------
@dataclass
class Pending:
    """Everything computable WITHOUT the network, plus the one question to ask.

    The split exists so that the per-sample entry point and the batched manager run
    identical arithmetic: `prepare` is pure, `finalize` is pure, and the only thing
    between them is whether the judge calls are made one at a time or in a pool.
    """
    route: str
    format_error: bool
    flags: dict
    answer_text: str
    zoomed: bool
    tool_rounds: int
    turns_missing: float
    #: (question, gold, answer) if a judge call is owed, else None
    ask: Optional[tuple] = None
    #: set when the answer is already decided offline
    acc: Optional[float] = None
    acc_source: int = ACC_JUDGE
    extra: dict = field(default_factory=dict)


def _route_of(data_source) -> str:
    ds = "" if data_source is None else str(data_source)
    if ds in MATH_DATA_SOURCES:
        return ROUTE_MATH
    if ds in REASONING_DATA_SOURCES:
        return ROUTE_REASONING
    if ds in VL_DATA_SOURCES:
        return ROUTE_VL
    raise NotImplementedError(
        f"DeepEyes reward is not implemented for data_source={ds!r}. Known: "
        f"{sorted(VL_DATA_SOURCES)} -> the tool composite, "
        f"{sorted(REASONING_DATA_SOURCES)} -> the tagged-answer composite, "
        f"{sorted(MATH_DATA_SOURCES)} -> the math composite. This mirrors their "
        "verl/utils/reward_score/__init__.py:64-74; add the string there and here "
        "together, and remember Prompt 6's trap: the ThinkLite split must have the "
        "tool DISABLED, or a quarter of the corpus gets the wrong composite.")


def prepare(data_source, solution_str, ground_truth, extra_info=None) -> Pending:
    """The offline half. No network, no torch, no model -- this is what the tests pin."""
    extra_info = extra_info or {}
    route = _route_of(data_source)
    question = extra_info.get("question") or ""
    flags = tag_flags(solution_str)
    zoomed, rounds, missing = zoom_from(extra_info.get("num_turns"), solution_str)

    if route == ROUTE_MATH:
        # compute_score_math: think balance only -- no vision clause, no answer clause.
        fmt_err = bool(flags["think_miscount"])
        tail = (solution_str or "").split("</think>")[-1].strip()
        found = _BOXED_THEIRS.findall(tail)
        # Compare the CONTENT of the last box, not the number of boxes: on
        # `\boxed{\frac{1}{2}}` both patterns find exactly one match, and theirs
        # truncates it to `\frac{1`. Counting matches would report no disagreement.
        bal = _BOXED_BALANCED.findall(tail)
        nested = float(bool(found) and bool(bal) and bal[-1] != found[-1])
        p = Pending(route=route, format_error=fmt_err, flags=flags, answer_text="",
                    zoomed=zoomed, tool_rounds=rounds, turns_missing=missing,
                    extra={"boxed_n": float(len(found)), "boxed_nested": nested})
        if not found:
            p.format_error = True
            p.acc, p.acc_source = 0.0, ACC_EMPTY
            return p
        if len(found) > 1:
            p.format_error = True
        p.answer_text = found[-1]
        if rule_math_verify(ground_truth, p.answer_text):
            p.acc, p.acc_source = 1.0, ACC_RULE
        else:
            p.ask = (question, ground_truth, p.answer_text)
        return p

    if route == ROUTE_REASONING:
        # compute_common_reasoning: the tagged parse, and a missing tag IS an error.
        fmt_err = bool(flags["think_miscount"] or flags["vision_miscount"]
                       or flags["answer_miscount"])
        answer = extract_answer_tagged(solution_str)
        p = Pending(route=route, format_error=fmt_err, flags=flags,
                    answer_text=answer or "", zoomed=zoomed, tool_rounds=rounds,
                    turns_missing=missing)
        if not answer:
            p.format_error = True
            p.acc, p.acc_source = 0.0, ACC_EMPTY
        elif len(answer) >= ANSWER_CHAR_LIMIT:
            p.format_error = True
            p.acc, p.acc_source = 0.0, ACC_EMPTY
        else:
            p.ask = (question, ground_truth, answer)
        return p

    # ROUTE_VL -- compute_score
    fmt_err = bool(flags["think_miscount"] or flags["vision_miscount"]
                   or flags["answer_miscount"])
    answer = extract_answer_vl(solution_str)
    p = Pending(route=route, format_error=fmt_err, flags=flags, answer_text=answer,
                zoomed=zoomed, tool_rounds=rounds, turns_missing=missing)
    if len(answer) >= ANSWER_CHAR_LIMIT:
        # their guard: the long answer is both wrong AND a format error, and it is
        # applied AFTER the judge in their code. Applying it before is the same number
        # and saves the call.
        p.format_error = True
        p.acc, p.acc_source = 0.0, ACC_EMPTY
    elif not answer:
        p.acc, p.acc_source = 0.0, ACC_EMPTY
    else:
        p.ask = (question, ground_truth, answer)
    return p


#: EVERY returned dict carries EVERY key, always. verl turns `reward_extra_info` into
#: `{k: np.array(v)}` on the batch's non_tensor_batch, and DataProto requires each of
#: those to have one entry per sample; a key that only the math route emits
#: (`boxed_n`) would make the batch ragged and blow up at a point that looks nothing
#: like the reward. The math-only fields are simply 0.0 elsewhere.
SCORE_SCHEMA: dict = {
    "score": 0.0,
    "acc": 0.0,
    "format": 0.0,
    "tool": 0.0,
    "zoomed": 0.0,
    "tool_rounds": 0.0,
    "turns_missing": 0.0,
    "judge_failed": 0.0,
    "judge_fallback": 0.0,
    "acc_source": 0.0,
    "answer_len": 0.0,
    "route_math": 0.0,
    "think_miscount": 0.0,
    "vision_miscount": 0.0,
    "answer_miscount": 0.0,
    "boxed_n": 0.0,
    "boxed_nested": 0.0,
    "reward_error": 0.0,
}


#: THE ALARM. Tier 2 is a safety net, and a safety net that holds the whole run is a
#: failure wearing a success's clothes: if the env vars never reached the ray actor,
#: every sample 401s, every sample falls through to the offline grader, and the run
#: completes with a plausible-looking reward curve graded by a substring match. The
#: per-sample `judge_fallback` metric is the real instrument; this is the thing that
#: shouts in the log so nobody has to be watching the right chart. Window, not total,
#: so a bad patch late in a run still trips it.
_ALARM_WINDOW = 256
_ALARM_FRACTION = 0.5
_alarm = {"n": 0, "degraded": 0, "fired": 0}
_alarm_lock = threading.Lock()


def _note_acc_source(source: int) -> None:
    with _alarm_lock:
        _alarm["n"] += 1
        if source in (ACC_FALLBACK, ACC_UNKNOWN):
            _alarm["degraded"] += 1
        if _alarm["n"] < _ALARM_WINDOW:
            return
        frac, fired = _alarm["degraded"] / _alarm["n"], _alarm["fired"]
        _alarm["n"] = _alarm["degraded"] = 0
        if frac <= _ALARM_FRACTION:
            return
        _alarm["fired"] = fired + 1
    print("\n" + "=" * 78 + "\n"
          f"[deepeyes-reward] ALARM: {frac:.0%} of the last {_ALARM_WINDOW} samples "
          "were NOT graded by the judge.\n"
          "  R_acc is coming from the offline string grader, which is weaker and will\n"
          "  quietly change what this arm optimises. Almost always this is the gateway:\n"
          "    - NVIDIA_API_KEY / OPENAI_BASE_URL / JUDGE_MODEL did not reach the ray\n"
          "      actor (runtime_env env_vars, not just the submitting shell), or\n"
          "    - the gateway is down / rate-limiting.\n"
          f"  Watch judge_fallback and judge_failed in the step metrics. (alarm #{fired + 1})\n"
          + "=" * 78 + "\n", flush=True)


def finalize(p: Pending, judged: Optional[float]) -> dict:
    """The composite, plus everything worth logging. `judged` is tier 1's verdict."""
    acc, source = p.acc, p.acc_source
    if acc is None:
        if judged is not None:
            acc, source = float(judged), ACC_JUDGE
        else:
            fallback = offline_grade(p.ask[2] if p.ask else "", p.ask[1] if p.ask else "")
            if fallback is not None:
                acc, source = fallback, ACC_FALLBACK
            else:
                acc, source = 0.0, ACC_UNKNOWN
                print("[deepeyes-reward] tier 3: neither the judge nor the offline "
                      f"grader could decide (gold={p.ask[1]!r}). Scoring R_acc=0.0 and "
                      "flagging judge_failed=1; R_tool cannot pay out on it.",
                      flush=True)

    _note_acc_source(source)
    fmt = FORMAT_PENALTY if p.format_error else 0.0
    if p.route == ROUTE_MATH:
        tool = 0.0
        score = MATH_W_ACC * acc + MATH_W_FORMAT * fmt
    else:
        # THE GATE. Both clauses, and nothing else.
        tool = 1.0 if (p.zoomed and acc > 0.5) else 0.0
        score = W_ACC * acc + W_FORMAT * fmt + W_TOOL * tool

    out = dict(SCORE_SCHEMA)
    out.update({
        "score": float(score),
        "acc": float(acc),                      # verl's validation picks "acc" as core
        "format": float(fmt),
        "tool": float(tool),
        "zoomed": float(p.zoomed),
        "tool_rounds": float(p.tool_rounds),
        "turns_missing": float(p.turns_missing),
        "judge_failed": 1.0 if source == ACC_UNKNOWN else 0.0,
        "judge_fallback": 1.0 if source == ACC_FALLBACK else 0.0,
        "acc_source": float(source),
        "answer_len": float(len(p.answer_text)),
        "route_math": 1.0 if p.route == ROUTE_MATH else 0.0,
    })
    out.update(p.flags)
    out.update(p.extra)
    assert out.keys() == SCORE_SCHEMA.keys(), f"schema drift: {out.keys() ^ SCORE_SCHEMA.keys()}"
    return out


# ---------------------------------------------------------------------------
# the entry points
# ---------------------------------------------------------------------------
def error_score() -> dict:
    """What a sample scores when the reward function itself broke.

    Same keys as `finalize`, so `reward_extra_info`'s lists stay rectangular -- verl
    asserts every extra key has one entry per sample and np.array()s them. 0.0 with
    `reward_error=1` rather than an exception, because a reward must not take a
    training step down; rather than NaN, because GRPO's advantage is a group mean/std
    and one NaN would destroy all 16 rollouts of the prompt instead of one.
    """
    out = dict(SCORE_SCHEMA)
    out.update({"judge_failed": 1.0, "acc_source": float(ACC_UNKNOWN), "reward_error": 1.0})
    return out


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None,
                  judge_fn: Optional[Callable] = None, **kwargs) -> dict:
    """verl's per-sample custom reward function. -> dict with a "score" key.

    `judge_fn(question, gold, answer) -> 1.0/0.0/None` is the seam the CPU test uses;
    verl never passes it and the gateway judge is the default.
    """
    try:
        p = prepare(data_source, solution_str, ground_truth, extra_info)
        judged = None
        if p.ask is not None:
            judged = (judge_fn or judge_binary)(*p.ask)
        return finalize(p, judged)
    except NotImplementedError:
        raise
    except Exception as e:  # a reward must never take a training step down with it
        print(f"[deepeyes-reward] UNHANDLED {type(e).__name__}: {str(e)[:200]} -- "
              "scoring 0.0 and flagging reward_error=1", flush=True)
        return error_score()


def score_batch(data_sources, solution_strs, ground_truths, extra_infos=None,
                workers: Optional[int] = None, judge_fn: Optional[Callable] = None):
    """The batched shape: prepare everything, pool the judge calls, finalize everything.

    Identical arithmetic to `compute_score` by construction -- same `prepare`, same
    `finalize`. Also the entry point for `reward_manager=batch`, whose convention is
    `compute_score(data_sources=, solution_strs=, ground_truths=, extra_infos=)`; note
    that THAT manager does not pass `__num_turns__` through, so it will log
    turns_missing=1 and fall back to the `<tool_call>` count. Prefer
    `reward_manager=deepeyes`.
    """
    n = len(solution_strs)
    extra_infos = list(extra_infos or [{} for _ in range(n)])
    pend = []
    for i in range(n):
        try:
            pend.append(prepare(data_sources[i], solution_strs[i], ground_truths[i],
                                extra_infos[i] or {}))
        except NotImplementedError:
            raise
        except Exception as e:
            print(f"[deepeyes-reward] prepare failed on item {i}: "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            pend.append(None)

    # DEDUPLICATED on the (question, gold, answer) triple before anything is asked --
    # `llm_judge.judge_scores`'s rule, and it bites harder here: GRPO draws 16 rollouts
    # of the SAME prompt, and on a short-answer corpus like V* many of them land on the
    # same string. Paying for each of those separately is money and rate limit for an
    # identical answer, and it would also let two rollouts with identical answers get
    # different R_acc from judge sampling noise, which is a spurious advantage.
    verdicts: dict[int, Optional[float]] = {}
    uniq: dict[tuple, list] = {}
    for i, p in enumerate(pend):
        if p is not None and p.ask is not None:
            uniq.setdefault(tuple(str(x) for x in p.ask), []).append(i)
    if uniq:
        fn = judge_fn or judge_binary
        w = workers if workers is not None else max(1, int(os.environ.get("JUDGE_MAX_WORKERS", "16")))

        def _ask(key):
            try:
                return fn(*key)
            except Exception as e:  # a judge_fn that raises must not kill the pool
                print(f"[deepeyes-reward] judge_fn raised: {type(e).__name__}: "
                      f"{str(e)[:140]}", flush=True)
                return None

        keys = list(uniq)
        with ThreadPoolExecutor(max_workers=min(w, len(keys))) as ex:
            for key, v in zip(keys, ex.map(_ask, keys)):
                for i in uniq[key]:
                    verdicts[i] = v

    out = []
    for i, p in enumerate(pend):
        out.append(error_score() if p is None else finalize(p, verdicts.get(i)))
    return out


#: `reward_manager=batch` calls `compute_score(data_sources=, solution_strs=, ...)`.
def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None,
                        **kwargs):
    return score_batch(data_sources, solution_strs, ground_truths, extra_infos)


# ---------------------------------------------------------------------------
# the batched reward manager
# ---------------------------------------------------------------------------
# Registered as a SIDE EFFECT of this module being loaded, which works because
# verl/trainer/ppo/reward.py:136 calls get_custom_reward_fn(config) -- importing this
# file -- before :148 resolves config.reward_model.reward_manager. If that order ever
# changes, the failure is a loud `ValueError: Unknown reward manager: deepeyes` at
# startup rather than a silent wrong number, which is the right way round.
#
# The registration is also made IDEMPOTENT. verl loads this file under the module name
# `custom_module`; anything else that imports it (a probe, a test, an eval script) gets
# a SECOND module object with a second, identical class, and `register` raises
# `ValueError: Reward manager deepeyes has already been registered` on a difference it
# does not care about. Re-use the registry's entry instead, so load order cannot
# decide whether a training job starts.
try:
    import torch
    from collections import defaultdict as _defaultdict

    from verl import DataProto
    from verl.workers.reward_manager.abstract import AbstractRewardManager
    from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY, register as _register

    class DeepEyesRewardManager(AbstractRewardManager):
        """NaiveRewardManager's contract, with the judge calls pooled.

        Two differences from `naive`, both deliberate:

        1. the judge calls for the whole batch go through one ThreadPoolExecutor. At
           256x16 the serial version is ~4,096 round trips per step.
        2. the response is decoded with `skip_special_tokens=False`, which is what
           DeepEyes' own naive.py:71 does. It keeps `<|vision_start|><|image_pad|>` in
           the string, so their vision-balance clause is live rather than inert -- see
           the module docstring. `num_turns` is still the rule for R_tool; this only
           restores the FORMAT clause.
        """

        def __init__(self, tokenizer, num_examine, compute_score=None,
                     reward_fn_key="data_source", **reward_kwargs):
            self.tokenizer = tokenizer
            self.num_examine = num_examine
            self.reward_fn_key = reward_fn_key
            self.workers = reward_kwargs.get("judge_workers")
            # `compute_score` arrives wrapped by get_custom_reward_fn and is the
            # per-sample shape; this manager batches instead, so it is kept only so a
            # misconfiguration is visible.
            self._custom = compute_score

        def __call__(self, data: "DataProto", return_dict: bool = False):
            if "rm_scores" in data.batch.keys():
                if return_dict:
                    keys = data.meta_info.get("reward_extra_keys", [])
                    return {"reward_tensor": data.batch["rm_scores"],
                            "reward_extra_info": {k: data.non_tensor_batch[k] for k in keys}}
                return data.batch["rm_scores"]

            prompt_ids = data.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            attn = data.batch["attention_mask"]
            valid_len = attn[:, prompt_len:].sum(dim=-1)

            n = len(data)
            solution_strs, data_sources, ground_truths, extras = [], [], [], []
            turns = data.non_tensor_batch.get("__num_turns__", None)
            for i in range(n):
                ids = data.batch["responses"][i][: valid_len[i]]
                solution_strs.append(self.tokenizer.decode(ids, skip_special_tokens=False))
                item = data[i]
                data_sources.append(item.non_tensor_batch[self.reward_fn_key])
                ground_truths.append(item.non_tensor_batch["reward_model"].get("ground_truth"))
                ei = dict(item.non_tensor_batch.get("extra_info") or {})
                ei["num_turns"] = None if turns is None else int(turns[i])
                extras.append(ei)

            scores = score_batch(data_sources, solution_strs, ground_truths, extras,
                                 workers=self.workers)

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = _defaultdict(list)
            printed: dict[Any, int] = {}
            for i in range(n):
                s = scores[i]
                reward_tensor[i, valid_len[i] - 1] = s["score"]
                for k, v in s.items():
                    reward_extra_info[k].append(v)
                ds = data_sources[i]
                if printed.get(ds, 0) < self.num_examine:
                    printed[ds] = printed.get(ds, 0) + 1
                    print("[prompt]", self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True))
                    print("[response]", solution_strs[i])
                    print("[ground_truth]", ground_truths[i])
                    print("[score]", s)

            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            return reward_tensor

    _already = REWARD_MANAGER_REGISTRY.get("deepeyes")
    if _already is None:
        _register("deepeyes")(DeepEyesRewardManager)
    else:
        DeepEyesRewardManager = _already

except ImportError:  # pragma: no cover - the CPU test runs without verl/torch
    DeepEyesRewardManager = None


if __name__ == "__main__":
    # Their own __main__ case, on their own numbers: two <answer> opens and two closes
    # in the tail balance, so format is 0; the parse takes the LAST <answer>, "left",
    # which is the gold. No tool turn -> no bonus. 0.8*1 + 0.2*0 + 1.2*0 = 0.8.
    demo = ("The answer is <think> 2 + 2 = 4 </think> <answer> right </answer> "
            "<answer> left </answer>")
    print(compute_score(
        data_source="vstar", solution_str=demo, ground_truth="left",
        extra_info={"question": "Is the woman to the left or to the right of the man?",
                    "num_turns": 2},
        judge_fn=lambda q, g, a: 1.0 if a.strip().lower() == str(g).lower() else 0.0))
