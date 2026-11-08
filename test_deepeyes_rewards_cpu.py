#!/usr/bin/env python
"""CPU checks for DeepEyes' reward composite. No GPU, no model, no gateway.

    python test_deepeyes_rewards_cpu.py

The judge is MOCKED everywhere -- the suite must not depend on the NVIDIA gateway
being up, and a reward test that costs money is a reward test nobody runs.

What each group is for:

  gate        the conditional tool bonus, which is the whole design: zoom + wrong
              answer pays nothing, no zoom + right answer pays nothing, and only both
              together pay 1.2. If this ever passes by accident the arm is a 1.2-point
              bonus for zooming, which the policy will farm on its first hundred steps.
  format      the -1 penalty and, just as important, the cases that are NOT penalised.
              "Any tag mismatch" is a COUNT comparison in their code, so a completion
              with no tags at all scores 0.0, not -1.0 -- see the module docstring.
  turns       num_turns -> tool rounds. `tool_agent_loop.py` emits
              user+assistant+1, so a no-tool rollout is 2 and each zoom adds 2. Reading
              that off by one makes every single-zoom rollout look tool-free.
  math        the thinklite route: 1.2*acc + 0.4*format, no tool term at all, boxed
              extraction, and the "more than one \\boxed is a format error" clause.
  judge       the three failure tiers. The one that matters is tier 2: a gateway that
              will not answer must not be read as "wrong", because the composite reads
              acc <= 0.5 as "suppress R_tool" and the run would quietly become plain
              GRPO with a penalty.
  schema      every route returns the same key set. verl np.array()s reward_extra_info
              onto the batch and a math-only key would make it ragged at a point in the
              trainer that looks nothing like the reward.
  batch       the pooled path is arithmetically identical to the serial one, and it
              deduplicates identical (question, gold, answer) triples.
  original    DIFFERENTIAL, against `vl_agent.py` from their own repo, with its network
              client stubbed. A transcription can only be checked against the original.
              Skipped with a note if the reference clone is absent; set DEEPEYES_REF to
              point at it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import deepeyes.rewards as DR  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def judge_const(v):
    """A judge that always returns the same verdict. `None` means "would not answer"."""
    return lambda q, g, a: v


EI = {"question": "What colour is the hydrant?"}


def ei(num_turns=2, question=EI["question"]):
    return {"question": question, "num_turns": num_turns}


#: a well-formed single-turn completion. Balanced <think>, balanced <answer>.
CLEAN = "<think> I look at the picture. </think> <answer> red </answer>"
#: the observation a tool turn splices in, as it appears when the caller decodes
#: WITHOUT skip_special_tokens. Qwen3-VL's ids: vision_start 151652, image_pad 151655,
#: vision_end 151653 -- identical to Qwen2.5-VL's, which is why their count trick is
#: about tokenisation and not about the model.
OBS = "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>"


def zoomed_completion(k=1):
    """k tool rounds, written the way the agent loop actually splices them."""
    body = ""
    for _ in range(k):
        body += ("<think> let me zoom </think>"
                 "<tool_call>{\"name\": \"image_zoom_in_tool\"}</tool_call>" + OBS)
    return body + CLEAN


# ---------------------------------------------------------------------------
def test_gate():
    print("\ngate -- the tool bonus pays only on a correct answer")
    right, wrong = judge_const(1.0), judge_const(0.0)

    no_zoom_right = DR.compute_score("vstar", CLEAN, "red", ei(2), judge_fn=right)
    zoom_right = DR.compute_score("vstar", zoomed_completion(1), "red", ei(4), judge_fn=right)
    zoom_wrong = DR.compute_score("vstar", zoomed_completion(1), "red", ei(4), judge_fn=wrong)
    no_zoom_wrong = DR.compute_score("vstar", CLEAN, "red", ei(2), judge_fn=wrong)

    check("zoom + correct answer scores 0.8 + 1.2 = 2.0",
          abs(zoom_right["score"] - 2.0) < 1e-12 and zoom_right["tool"] == 1.0,
          f"{zoom_right['score']:.3f}")
    check("zoom + WRONG answer gets no tool bonus (0.0, not 1.2)",
          abs(zoom_wrong["score"] - 0.0) < 1e-12 and zoom_wrong["tool"] == 0.0,
          f"{zoom_wrong['score']:.3f}")
    check("NO zoom + correct answer gets no tool bonus (0.8)",
          abs(no_zoom_right["score"] - 0.8) < 1e-12 and no_zoom_right["tool"] == 0.0,
          f"{no_zoom_right['score']:.3f}")
    check("no zoom + wrong answer is 0.0", abs(no_zoom_wrong["score"]) < 1e-12)
    check("the tool term is worth exactly 1.2, and only both clauses earn it",
          abs((zoom_right["score"] - no_zoom_right["score"]) - 1.2) < 1e-12
          and zoom_wrong["score"] == no_zoom_wrong["score"],
          f"delta {zoom_right['score'] - no_zoom_right['score']:.3f}")
    # The gate's threshold and the judge's binarisation have to be the same number, or
    # a judge score that rounds to exactly 0.5 would earn accuracy credit and no tool
    # bonus -- a state their composite has no name for.
    check("the gate threshold is the judge binarisation cut",
          DR.JUDGE_BINARISE_AT == 0.5)
    half = DR.compute_score("vstar", zoomed_completion(1), "red", ei(4),
                            judge_fn=judge_const(0.5))
    check("an acc of exactly 0.5 does NOT open the gate",
          half["tool"] == 0.0, f"score {half['score']:.3f}")


def test_format():
    print("\nformat -- -1 on a MISCOUNT, and 0 on a model that simply has no tags")
    right = judge_const(1.0)

    unbalanced = "<think> thinking <answer> red </answer>"       # no </think>
    r = DR.compute_score("vstar", unbalanced, "red", ei(2), judge_fn=right)
    check("an unclosed <think> is -1 -> 0.8 + 0.2*(-1) = 0.6",
          abs(r["score"] - 0.6) < 1e-12 and r["format"] == -1.0, f"{r['score']:.3f}")

    open_answer = "<think> t </think> <answer> red"
    r = DR.compute_score("vstar", open_answer, "red", ei(2), judge_fn=right)
    check("an unclosed <answer> after </think> is -1",
          r["format"] == -1.0 and r["answer_miscount"] == 1.0)

    tagless = "The hydrant is red."
    r = DR.compute_score("vstar", tagless, "red", ei(2), judge_fn=right)
    check("a TAGLESS completion is not penalised -- 0 == 0 on every clause",
          r["format"] == 0.0 and abs(r["score"] - 0.8) < 1e-12,
          "this is what Prompt 4 reads the wrong way round")

    long_answer = "<think> t </think> <answer> " + "x" * DR.ANSWER_CHAR_LIMIT + " </answer>"
    r = DR.compute_score("vstar", long_answer, "red", ei(4), judge_fn=right)
    check("the >=1000-char hack guard zeroes acc AND sets -1 AND kills the tool bonus",
          r["acc"] == 0.0 and r["format"] == -1.0 and r["tool"] == 0.0
          and abs(r["score"] + 0.2) < 1e-12, f"{r['score']:.3f}")

    # The tagless + long-chain case is the REAL cold-start argument: with no tags the
    # answer parse returns the whole completion, so a long chain trips the hack guard.
    tagless_long = "I think about it. " * 100
    r = DR.compute_score("vstar", tagless_long, "red", ei(2), judge_fn=right)
    check("a tagless LONG chain does get -1, via the answer parse not the tag counts",
          r["format"] == -1.0 and r["acc"] == 0.0, f"answer_len {r['answer_len']:.0f}")

    # The vision clause. Live when the caller keeps special tokens, inert when it does
    # not -- which is this fork's naive manager. Both states are asserted so that
    # "it never fires" stays a measurement.
    torn = "<think> t </think>" + "<|vision_start|><|image_pad|>" + "<answer> red </answer>"
    r = DR.compute_score("vstar", torn, "red", ei(4), judge_fn=right)
    check("an unbalanced vision splice is -1 when special tokens are in the string",
          r["vision_miscount"] == 1.0 and r["format"] == -1.0)
    stripped = torn.replace("<|vision_start|>", "").replace("<|image_pad|>", "")
    r = DR.compute_score("vstar", stripped, "red", ei(4), judge_fn=right)
    check("...and is INERT once skip_special_tokens=True has removed them",
          r["vision_miscount"] == 0.0 and r["format"] == 0.0)


def test_turns():
    print("\nturns -- num_turns is the zoom signal, not the token count")
    for n, want in [(2, 0), (3, 1), (4, 1), (5, 2), (6, 2), (8, 3)]:
        check(f"num_turns={n} -> {want} tool round(s)",
              DR.tool_rounds_from_turns(n) == want)
    check("num_turns=2 is NOT a zoom (one generate round, no tool)",
          DR.zoom_from(2, "")[0] is False)
    check("num_turns=4 IS a zoom", DR.zoom_from(4, "")[0] is True)

    # The fallback must be announced, never silent: a run whose R_tool is secretly
    # always 0 looks exactly like a run whose policy never learned to zoom.
    zoomed, k, missing = DR.zoom_from(None, zoomed_completion(2))
    check("a missing num_turns falls back to <tool_call> AND raises turns_missing",
          zoomed is True and k == 2 and missing == 1.0)
    zoomed, k, missing = DR.zoom_from(None, CLEAN)
    check("the fallback reports no zoom on a tool-free completion, still flagged",
          zoomed is False and k == 0 and missing == 1.0)
    check("a non-integer num_turns is treated as missing, not as 0",
          DR.zoom_from("junk", zoomed_completion(1))[2] == 1.0)

    # The thing the fallback cannot do, and the reason it is the fallback.
    faked = CLEAN + "<tool_call>I am typing this, I did not call anything</tool_call>"
    check("a model TYPING <tool_call> fools the fallback but not num_turns",
          DR.zoom_from(None, faked)[0] is True and DR.zoom_from(2, faked)[0] is False)


def test_math():
    print("\nmath -- 1.2*acc + 0.4*format, and no tool term at all")
    boxed = "<think> 2+2 </think> The answer is \\boxed{4}"
    r = DR.compute_score("thinklite_eureka", boxed, "4", ei(2), judge_fn=judge_const(0.0))
    check("a rule-verifiable \\boxed{4} against gold 4 scores 1.2 without a judge call",
          abs(r["score"] - 1.2) < 1e-12 and r["acc_source"] == float(DR.ACC_RULE),
          f"{r['score']:.3f}" + ("" if DR.HAVE_MATH_VERIFY else "  [no math_verify here]"))

    r = DR.compute_score("thinklite_eureka", "<think> 2+2 </think> \\boxed{5}", "4",
                         ei(2), judge_fn=judge_const(0.0))
    check("a wrong box the generative verifier also rejects scores 0.0",
          abs(r["score"]) < 1e-12 and r["acc"] == 0.0, f"{r['score']:.3f}")

    r = DR.compute_score("thinklite_eureka", "<think> 2+2 </think> four", "4",
                         ei(2), judge_fn=judge_const(1.0))
    check("no \\boxed at all is acc 0 AND format -1 -> -0.4",
          abs(r["score"] + 0.4) < 1e-12 and r["format"] == -1.0, f"{r['score']:.3f}")

    r = DR.compute_score("thinklite_eureka", "<think> t </think> \\boxed{3} \\boxed{4}",
                         "4", ei(2), judge_fn=judge_const(0.0))
    check("two boxes: the LAST one is graded, and it is a format error -> 1.2-0.4=0.8",
          abs(r["score"] - 0.8) < 1e-12 and r["boxed_n"] == 2.0, f"{r['score']:.3f}")

    # The term that must not exist on this route.
    with_zoom = DR.compute_score("thinklite_eureka", "<think> t </think> \\boxed{4}",
                                 "4", ei(8), judge_fn=judge_const(1.0))
    check("the math route pays NO tool bonus even on 3 zooms (Prompt 6's trap)",
          with_zoom["tool"] == 0.0 and abs(with_zoom["score"] - 1.2) < 1e-12,
          f"{with_zoom['score']:.3f}")

    # Their regex is `\\boxed\{([^}]+)\}`, which truncates a nested expression. We keep
    # their regex (fidelity) and COUNT the disagreement instead of silently differing.
    nested = "<think> t </think> \\boxed{\\frac{1}{2}}"
    r = DR.compute_score("thinklite_eureka", nested, "0.5", ei(2), judge_fn=judge_const(1.0))
    check("a nested \\boxed is flagged boxed_nested rather than quietly mis-parsed",
          r["boxed_nested"] == 1.0)

    check("the math route is reached only by their data_source strings",
          DR._route_of("thinklite_eureka") == DR.ROUTE_MATH
          and DR._route_of("xince") == DR.ROUTE_MATH
          and DR._route_of("vstar") == DR.ROUTE_VL
          and DR._route_of("chart") == DR.ROUTE_VL
          and DR._route_of("geoguessr") == DR.ROUTE_REASONING)
    try:
        DR.compute_score("hiyouga/geometry3k", CLEAN, "red", ei(2), judge_fn=judge_const(1.0))
        check("an unknown data_source raises rather than guessing a composite", False)
    except NotImplementedError:
        check("an unknown data_source raises rather than guessing a composite", True)


def test_judge_failure():
    print("\njudge -- the three failure tiers, and what each one must NOT do")
    # tier 2: the gateway would not answer. The offline grader can still read "red".
    r = DR.compute_score("vstar", CLEAN, "red", ei(4), judge_fn=judge_const(None))
    check("a judge that will not answer falls back to the offline grader, not to 0",
          r["acc"] == 1.0 and r["acc_source"] == float(DR.ACC_FALLBACK)
          and r["judge_fallback"] == 1.0 and r["judge_failed"] == 0.0)
    check("...and the tool bonus still pays out on the fallback's verdict",
          abs(r["score"] - 2.0) < 1e-12, f"{r['score']:.3f}")

    wrong = "<think> t </think> <answer> blue </answer>"
    r = DR.compute_score("vstar", wrong, "red", ei(4), judge_fn=judge_const(None))
    check("the fallback can also say WRONG, and then the gate closes",
          r["acc"] == 0.0 and r["tool"] == 0.0 and abs(r["score"]) < 1e-12)

    # tier 3: nothing can decide, because the row has no ground truth at all.
    r = DR.compute_score("vstar", CLEAN, "", ei(4), judge_fn=judge_const(None))
    check("no judge and no gold is tier 3: acc 0, judge_failed=1, and it is PRINTED",
          r["acc"] == 0.0 and r["judge_failed"] == 1.0
          and r["acc_source"] == float(DR.ACC_UNKNOWN))
    check("tier 3 never returns NaN -- verl's GRPO advantage is a group mean/std",
          r["score"] == r["score"])

    # empty answer: 0.0 with no call at all. `llm_judge.judge_scores`'s rule.
    called = []
    r = DR.compute_score("vstar", "", "red", ei(4),
                         judge_fn=lambda *a: called.append(a) or 1.0)
    check("an empty completion is 0.0 with no judge call spent on it",
          r["acc"] == 0.0 and not called and r["acc_source"] == float(DR.ACC_EMPTY))

    # and the whole thing must never raise into the training step.
    r = DR.compute_score("vstar", CLEAN, "red", ei(4),
                         judge_fn=lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    check("a judge_fn that RAISES is caught and reported, not propagated",
          r["reward_error"] == 1.0 and r["score"] == 0.0)


def test_schema():
    print("\nschema -- every route returns the same keys, in every outcome")
    outs = [
        DR.compute_score("vstar", CLEAN, "red", ei(4), judge_fn=judge_const(1.0)),
        DR.compute_score("chart", "no tags here", "red", ei(2), judge_fn=judge_const(0.0)),
        DR.compute_score("geoguessr", CLEAN, "red", ei(4), judge_fn=judge_const(1.0)),
        DR.compute_score("thinklite_eureka", "<think> t </think> \\boxed{4}", "4",
                         ei(2), judge_fn=judge_const(1.0)),
        DR.compute_score("vstar", CLEAN, "", ei(2), judge_fn=judge_const(None)),
        DR.error_score(),
    ]
    keys = {frozenset(o) for o in outs}
    check("all routes and both error paths share one key set", len(keys) == 1,
          f"{len(next(iter(keys)))} keys")
    check("every value is a float -- reward_extra_info is np.array()d and averaged",
          all(isinstance(v, float) for o in outs for v in o.values()))
    check('the accuracy key is named "acc", which verl picks as the core val metric',
          "acc" in outs[0])


def test_reasoning_route():
    print("\nreasoning -- geoguessr's parse differs: a MISSING <answer> is an error")
    r = DR.compute_score("geoguessr", "The hydrant is red.", "red", ei(4),
                         judge_fn=judge_const(1.0))
    check("no <answer> tags on this route is acc 0 and format -1",
          r["acc"] == 0.0 and r["format"] == -1.0, f"{r['score']:.3f}")
    r = DR.compute_score("geoguessr", CLEAN, "red", ei(4), judge_fn=judge_const(1.0))
    check("a tagged answer on this route takes the full composite, tool included",
          abs(r["score"] - 2.0) < 1e-12, f"{r['score']:.3f}")


def test_batch():
    print("\nbatch -- the pooled path is the serial path, and it deduplicates")
    cases = [
        ("vstar", zoomed_completion(1), "red", ei(4)),
        ("vstar", CLEAN, "red", ei(2)),
        ("vstar", zoomed_completion(1), "red", ei(4)),       # duplicate triple
        ("chart", "<think> t </think> <answer> blue </answer>", "red", ei(4)),
        ("thinklite_eureka", "<think> t </think> \\boxed{4}", "4", ei(2)),
    ]
    seen = []

    def spy(q, g, a):
        seen.append((q, g, a))
        return 1.0 if str(a).strip().lower() == str(g).strip().lower() else 0.0

    batched = DR.score_batch([c[0] for c in cases], [c[1] for c in cases],
                             [c[2] for c in cases], [c[3] for c in cases],
                             workers=4, judge_fn=spy)
    n_calls = len(seen)
    serial = [DR.compute_score(*c, judge_fn=spy) for c in cases]
    check("batched == serial, key by key", batched == serial)
    # 5 rows -> 2 judge calls. Rows 0, 1 and 2 all ask ("...hydrant?", "red", "red"):
    # one of them zoomed and one did not, but that changes R_tool, not the question
    # the judge is asked. Row 3 answers "blue" (a second triple) and row 4 is
    # rule-verified offline. Deduplicating is money AND correctness -- two rollouts
    # with the identical answer must not get different R_acc from sampling noise.
    want = 2 if DR.HAVE_MATH_VERIFY else 3
    check("identical (question, gold, answer) triples are asked once",
          n_calls == want, f"{n_calls} judge calls for 5 rows (want {want})")
    check("the batch's composites are the expected ones",
          abs(batched[0]["score"] - 2.0) < 1e-12
          and abs(batched[1]["score"] - 0.8) < 1e-12
          and abs(batched[3]["score"] - 0.0) < 1e-12,
          f"{[round(b['score'], 2) for b in batched]}")

    # A prepare() that explodes must cost one row, not the step.
    broken = DR.score_batch(["vstar"], [None], ["red"], [{"question": 1}],
                            workers=1, judge_fn=spy)
    check("a row that cannot be prepared scores 0 with reward_error=1",
          len(broken) == 1 and broken[0]["score"] == 0.0)


# ---------------------------------------------------------------------------
def _load_their_vl_agent():
    """Import DeepEyes' own vl_agent.py with its network client stubbed out.

    Their module builds an OpenAI client AND calls `requests.get(f"{base}/models")` at
    import time, so it cannot be imported as-is on a login node. The stubs replace both
    and nothing else -- every line of scoring logic is theirs.
    """
    ref = os.environ.get("DEEPEYES_REF", "/home/uberger/scratch/DeepEyes")
    path = os.path.join(ref, "verl", "utils", "reward_score", "vl_agent.py")
    if not os.path.exists(path):
        return None, ref

    class _Create:
        fn = staticmethod(lambda **kw: None)

        def create(self, **kw):
            return type(self).fn(**kw)

    class _FakeClient:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=_Create())

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    fake_requests = types.ModuleType("requests")
    fake_requests.get = lambda url: types.SimpleNamespace(
        json=lambda: {"data": [{"id": "stub-judge"}]})

    saved = {k: sys.modules.get(k) for k in ("openai", "requests")}
    sys.modules["openai"], sys.modules["requests"] = fake_openai, fake_requests
    try:
        spec = importlib.util.spec_from_file_location("their_vl_agent", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, path


def _reply(text):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])


def test_against_original():
    print("\noriginal -- differential against Visual-Agent/DeepEyes' vl_agent.py")
    them, where = _load_their_vl_agent()
    if them is None:
        print(f"  SKIP  reference clone not found at {where}")
        print("        git clone --depth 1 https://github.com/Visual-Agent/DeepEyes "
              "$HOME/scratch/DeepEyes   (or set DEEPEYES_REF)")
        return

    # The two "a zoom happened" signals have to agree BY CONSTRUCTION or the
    # differential measures the substitution instead of the transcription. So k is
    # derived from the string with THEIR rule -- `count("<|vision_start|><|image_pad|>")`
    # -- and fed to ours as num_turns = 2 + 2k. The one case where the two genuinely
    # disagree is asserted separately at the end of this function.
    cases = [
        ("clean, no zoom", CLEAN, "red"),
        ("clean, one zoom", zoomed_completion(1), "red"),
        ("clean, two zooms", zoomed_completion(2), "red"),
        ("unclosed think", "<think> t <answer> red </answer>", "red"),
        ("unclosed answer", OBS + "<think> t </think> <answer> red", "red"),
        ("no tags at all", "The hydrant is red.", "red"),
        ("no tags, one zoom", OBS + "The hydrant is red.", "red"),
        ("two answer blocks",
         OBS + "<think> t </think> <answer> a </answer> <answer> red </answer>", "red"),
        ("1000-char answer",
         OBS + OBS + "<think> t </think> <answer> " + "x" * 1000 + " </answer>", "red"),
        ("torn vision splice", "<think> t </think>" + "<|vision_start|><|image_pad|>"
         + "<answer> red </answer>", "red"),
    ]

    bad = []
    for label, text, gold in cases:
        k = text.count("<|vision_start|><|image_pad|>")
        for verdict in (1.0, 0.0):
            type(them.client_list[0].chat.completions).fn = staticmethod(
                lambda _v=verdict, **kw: _reply("1" if _v else "0"))
            theirs = them.compute_score(text, gold, {"question": EI["question"]})
            ours = DR.compute_score("vstar", text, gold,
                                    {"question": EI["question"], "num_turns": 2 + 2 * k},
                                    judge_fn=judge_const(verdict))["score"]
            if abs(theirs - ours) > 1e-12:
                bad.append(f"{label} @acc={verdict}: theirs {theirs:.3f} vs ours {ours:.3f}")
    check(f"compute_score matches theirs on {len(cases)} cases x 2 verdicts",
          not bad, "; ".join(bad[:3]))

    math_cases = [
        ("rule-verifiable", "<think> t </think> \\boxed{4}", "4"),
        ("wrong box", "<think> t </think> \\boxed{5}", "4"),
        ("no box", "<think> t </think> four", "4"),
        ("two boxes", "<think> t </think> \\boxed{3} \\boxed{4}", "4"),
        ("unclosed think", "<think> t \\boxed{4}", "4"),
    ]
    bad = []
    for label, text, gold in math_cases:
        for verdict in (1.0, 0.0):
            type(them.client_list[0].chat.completions).fn = staticmethod(
                lambda _v=verdict, **kw: _reply(
                    "## Equivalence Judgement\n" + ("TRUE" if _v else "FALSE")))
            theirs = them.compute_score_math(text, gold, {"question": EI["question"]})
            ours = DR.compute_score("thinklite_eureka", text, gold,
                                    {"question": EI["question"], "num_turns": 2},
                                    judge_fn=judge_const(verdict))["score"]
            if abs(theirs - ours) > 1e-12:
                bad.append(f"{label} @acc={verdict}: theirs {theirs:.3f} vs ours {ours:.3f}")
    check(f"compute_score_math matches theirs on {len(math_cases)} cases x 2 verdicts",
          not bad, "; ".join(bad[:3]))

    # The one place we deliberately DIVERGE, asserted so it cannot drift into an
    # accident: their R_tool reads the response string, ours reads num_turns. Under
    # this fork's `skip_special_tokens=True` decode their signal is gone.
    stripped = zoomed_completion(1).replace("<|vision_start|>", "") \
                                   .replace("<|image_pad|>", "") \
                                   .replace("<|vision_end|>", "")
    type(them.client_list[0].chat.completions).fn = staticmethod(lambda **kw: _reply("1"))
    theirs = them.compute_score(stripped, "red", {"question": EI["question"]})
    ours = DR.compute_score("vstar", stripped, "red",
                            {"question": EI["question"], "num_turns": 4},
                            judge_fn=judge_const(1.0))["score"]
    check("on a skip_special_tokens decode THEIR R_tool dies and ours survives",
          abs(theirs - 0.8) < 1e-12 and abs(ours - 2.0) < 1e-12,
          f"theirs {theirs:.3f}, ours {ours:.3f}")


def test_wiring():
    print("\nwiring -- the shapes verl will call")
    check("compute_score takes verl's keyword signature",
          DR.compute_score(data_source="vstar", solution_str=CLEAN, ground_truth="red",
                           extra_info=ei(4), judge_fn=judge_const(1.0))["score"] == 2.0)
    # An empty completion needs no judge, so this exercises the signature without
    # touching the gateway -- the suite must stay runnable with no key in the shell.
    check("compute_score_batch takes the BatchRewardManager signature",
          len(DR.compute_score_batch(["vstar"], [""], ["red"], [ei(2)])) == 1)
    have_verl = DR.DeepEyesRewardManager is not None
    print(f"  {'PASS' if have_verl else 'SKIP'}  the 'deepeyes' reward manager "
          f"{'registered' if have_verl else 'needs verl+torch (run in the deepeyes env)'}")
    if have_verl:
        from verl.workers.reward_manager import get_reward_manager_cls
        check("get_reward_manager_cls('deepeyes') resolves",
              get_reward_manager_cls("deepeyes") is DR.DeepEyesRewardManager)


if __name__ == "__main__":
    print(f"deepeyes.rewards from {DR.__file__}")
    print(f"math_verify available: {DR.HAVE_MATH_VERIFY}")
    test_gate()
    test_format()
    test_turns()
    test_math()
    test_judge_failure()
    test_schema()
    test_reasoning_route()
    test_batch()
    test_wiring()
    test_against_original()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        sys.exit(1)
    print("all checks passed")
