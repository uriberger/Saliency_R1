"""CPU test: real FLAN-T5 classifier load + segment_observe_steps token mapping."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

ost = _load("overlap_steps", "trl_repo/trl/trainer/overlap_steps.py")

# --- real classifier loads and predicts POD labels ---
clf = ost.OverlapStepsClassifier.load(device="cpu")
for s in ["I see a red car on the left side of the image.",
          "Therefore, the answer must be option B.",
          "Let me plan how to solve this problem."]:
    print(f"  predict({s[:40]!r:44}) -> {clf.predict(s, s, 'What color is the car?')}")
print("[T5] classifier load + predict OK")

# --- segment_observe_steps with a whitespace-token mock `out` ---
class _MockOut:
    """char_to_token via a simple whitespace tokenization of one string."""
    def __init__(self, text):
        self.spans = []  # (char_start, char_end, token_idx)
        tok = 0
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            self.spans.append((i, j, tok)); tok += 1
            i = j
    def char_to_token(self, case_id, char):
        for cs, ce, t in self.spans:
            if cs <= char < ce:
                return t
        return None

# force the classifier to select the middle sentence deterministically
observe_sentence = "I can see a red car in the picture."
text = f"<think> First I think. {observe_sentence} So the answer is A. </think> A"
out = _MockOut(text)
# token indices of the think content region
ts = out.char_to_token(0, text.index("First"))
te = out.char_to_token(0, text.index("A.") + 1)
think_start_char = text.index("First")
think_end_char = text.index("A.") + 1

class _FakeClf:
    def predict(self, step_text, chain, question):
        return "observe" if "see" in step_text else "deduce"

steps = ost.segment_observe_steps(text, think_start_char, think_end_char, out, 0, ts, te, "q", _FakeClf())
assert len(steps) == 1, steps
step_text, tok_a, tok_b = steps[0]
assert "see" in step_text
# token span must fall within [ts, te+1] and cover the observe sentence's words
assert ts <= tok_a < tok_b <= te + 1, (ts, te, tok_a, tok_b)
covered = " ".join(w for cs, ce, t in out.spans if tok_a <= t < tok_b for w in [text[cs:ce]])
print(f"  observe step tokens [{tok_a},{tok_b}) -> {covered!r}")
assert "see" in covered and "car" in covered
print("[T6] segment_observe_steps token mapping OK")
print("\nAll steps/classifier tests passed.")
