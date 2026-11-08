#!/usr/bin/env python
"""CPU tests for the ERGO low-resolution gate.

Everything here runs on a login node in seconds. What it is guarding is not the
arithmetic -- it is the three ways this probe could produce a confident number
that means nothing:

1. The cap not being applied at all, because `--force_simple` was dropped and the
   chat wrapper silently ignores `max_pixels`. Then every arm is the same arm and
   the gap reads 0.0, which is also the "stop the port" answer.
2. The cap being applied with the wrong factor. 28 instead of 32 is a 23% error in
   the token budget, in the direction that makes the penalty look larger.
3. HR-Bench's scorer reaching for the GPT judge, which on a compute node means ten
   dead POSTs per item and then a KeyError that discards the task.

Run: python test_ergo_gate_lowres_cpu.py
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import ergo_gate_lowres as gate  # noqa: E402


def load_generated(task_dir, name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, Path(task_dir) / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TokenBudget(unittest.TestCase):
    def test_factor_is_32_not_28(self):
        # Qwen3-VL is patch-16 / merge-2. If this ever reads 28*28 the caps are
        # 23% smaller than the labels on the table say they are.
        self.assertEqual(gate.VISUAL_TOKEN_PX, 32 * 32)

    def test_cap_reaches_the_processor_in_the_simple_path(self):
        """The cap must actually shrink the image, and only the simple path does it.

        This is the measurement the whole gate rests on, so it is checked against
        qwen_vl_utils itself rather than against our own arithmetic.
        """
        from qwen_vl_utils.vision_process import smart_resize

        for cap in gate.CAPS:
            height, width = smart_resize(1500, 2000, factor=32,
                                         min_pixels=gate.MIN_PIXELS,
                                         max_pixels=cap * gate.VISUAL_TOKEN_PX)
            tokens = height * width // gate.VISUAL_TOKEN_PX
            self.assertLessEqual(tokens, cap, f"cap {cap} overshot at {tokens} tokens")
            # Rounding down to whole 32-px steps costs at most one row and one
            # column, never a meaningful fraction of the budget.
            if cap * gate.VISUAL_TOKEN_PX < 2000 * 1500:
                self.assertGreater(tokens, 0.9 * cap, f"cap {cap} undershot at {tokens}")

    def test_native_resolution_caps_the_top_arm(self):
        """V*'s photos cannot reach 16384 tokens, and the report must not pretend they do.

        An image is never enlarged to meet the cap, so the 16384 arm on V* is
        "native resolution", not "16384 tokens". Reporting it as the latter would
        overstate the top of the curve.
        """
        from qwen_vl_utils.vision_process import smart_resize

        height, width = smart_resize(1500, 2000, factor=32, min_pixels=gate.MIN_PIXELS,
                                     max_pixels=16384 * gate.VISUAL_TOKEN_PX)
        self.assertLess(height * width // gate.VISUAL_TOKEN_PX, 4000)


class Command(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_force_simple_is_always_passed(self):
        for unit in gate.plan():
            cmd = gate.unit_command(unit, self.tmp, 8)
            self.assertIn("--force_simple", cmd,
                          f"{unit['id']}: without it max_pixels is inert and every arm is identical")

    def test_max_pixels_is_in_model_args_and_matches_the_cap(self):
        for unit in gate.plan():
            cmd = gate.unit_command(unit, self.tmp, 8)
            model_args = cmd[cmd.index("--model_args") + 1]
            self.assertIn(f"max_pixels={unit['cap'] * gate.VISUAL_TOKEN_PX}", model_args)
            self.assertIn(f"min_pixels={gate.MIN_PIXELS}", model_args)

    def test_no_response_cache(self):
        """--use_cache keys on (context, gen_kwargs, doc_id) and NOT on the pixel cap.

        Sharing one across caps would serve the 640-token answers back for every
        arm, and the gap would read exactly 0.0 -- the same output as a real null.
        """
        for unit in gate.plan():
            self.assertNotIn("--use_cache", gate.unit_command(unit, self.tmp, 8))

    def test_hrbench_never_logs_samples(self):
        """Its image column is a base64 STRING, which lmms-eval's media filter misses.

        saved_doc drops PIL/bytes/ndarray values; a str is none of those, so an 8K
        JPEG would be written into the samples jsonl once per row.
        """
        for unit in gate.plan():
            if any("hrbench" in t for t in unit["tasks"]):
                self.assertFalse(unit["log_samples"], unit["id"])

    def test_every_task_has_a_metric_and_a_label(self):
        for unit in gate.plan():
            for task in unit["tasks"]:
                self.assertIn(task, gate.TASK_METRIC)
                self.assertIn(task, gate.TASK_LABEL)

    def test_unit_ids_are_unique(self):
        ids = [u["id"] for u in gate.plan()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_grid_is_the_grid_the_gate_asked_for(self):
        units = gate.plan()
        combos = {(u["model"], t, u["cap"]) for u in units for t in u["tasks"]}
        for model in ("base", "coldstart"):
            for cap in (640, 1280, 16384):
                for task in ("vstar_bench", "hrbench4k_gate", "hrbench8k_gate"):
                    self.assertIn((model, task, cap), combos)
            self.assertIn((model, "vstar_bench_gtcrop", 640), combos)


class GeneratedTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.task_dir = gate.make_tasks(cls.tmp, hrbench_n=7, seed=99)

    def test_yaml_parses_and_resolves_its_functions(self):
        """Parse exactly as lmms-eval does, including the `!function` constructor.

        A generated config that loads here cannot fail at task-registration time
        for a reason this probe introduced -- which on an 8-rank job surfaces 2
        minutes in, after the model has loaded, once per unit.
        """
        from lmms_eval.utils import load_yaml_config

        for name in ("hrbench4k_gate", "hrbench8k_gate", "vstar_bench_gtcrop"):
            config = load_yaml_config(yaml_path=str(Path(self.task_dir) / f"{name}.yaml"))
            self.assertEqual(config["task"], name)
            self.assertTrue(callable(config["doc_to_visual"]))
            self.assertTrue(callable(config["process_results"]))

    def test_hrbench_scorer_never_calls_the_judge(self):
        module = load_generated(self.task_dir, "hrbench_gate")
        doc = {"index": 1, "cycle_category": "0", "category": "single", "answer": "B",
               "question": "q", "A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}

        calls = []
        module.hrbench_evaluator._post_request = lambda payload: calls.append(payload)

        for prediction, expect_score, expect_fail in [
            ("B", 1, 0),
            ("The answer is B.", 1, 0),
            ("<think>Maybe A, maybe C.</think>\nB. beta", 1, 0),
            ("<think>It is clearly B.</think>\nA", 0, 0),
            (r"\boxed{B}", 1, 0),
            ("beta", 1, 0),
            ("", 0, 1),
            ("I cannot tell from this image.", 0, 1),
        ]:
            out = module.process_results(doc, [prediction])
            self.assertEqual(out["average"]["gpt_score"], expect_score, prediction)
            self.assertEqual(out["parse_fail"], expect_fail, prediction)
        self.assertEqual(calls, [], "the GPT judge was reached; on a compute node that is a KeyError")

    def test_hrbench_scorer_reads_the_answer_not_the_thinking(self):
        """The trap the LogicVista and MathVision artifacts were both instances of."""
        module = load_generated(self.task_dir, "hrbench_gate")
        doc = {"index": 1, "cycle_category": "0", "category": "single", "answer": "D",
               "question": "q", "A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}
        out = module.process_results(
            doc, ["<think>At first glance the answer is A, but on a closer look</think>\nD"])
        self.assertEqual(out["average"]["gpt_score"], 1)

    def test_hrbench_aggregate_averages_over_cycle_categories(self):
        module = load_generated(self.task_dir, "hrbench_gate")
        rows = [{"index": 0, "cycle_category": "0", "gpt_score": 1},
                {"index": 1, "cycle_category": "0", "gpt_score": 0},
                {"index": 2, "cycle_category": "1", "gpt_score": 1}]
        self.assertAlmostEqual(module.aggregate(rows, None), (0.5 + 1.0) / 2)
        self.assertAlmostEqual(module.aggregate_parse_fail([0, 0, 1, 1], None), 0.5)

    def test_sampler_is_prefix_stable(self):
        """A 7-item sample must be a prefix of a 300-item one, or old rows move."""
        import datasets

        data = datasets.Dataset.from_dict({"i": list(range(50))})
        small = load_generated(self.task_dir, "gatesample").take(data)
        big = gate.make_tasks(tempfile.mkdtemp(), hrbench_n=20, seed=99)
        bigger = load_generated(big, "gatesample").take(data)
        self.assertEqual(small["i"], bigger["i"][:7])

    def test_gtcrop_crops_the_ground_truth_region(self):
        from PIL import Image

        module = load_generated(self.task_dir, "gtcrop")
        boxes = json.loads((REPO / "assets" / "vstar_bench_boxes.json").read_text())
        qid, entry = next(iter(sorted(boxes.items(), key=lambda kv: int(kv[0]))))

        image = Image.new("RGB", (2000, 1500), "white")
        visuals = module.doc_to_visual({"image": image, "question_id": qid})
        self.assertEqual(len(visuals), 2, "the crop arm must send TWO images")
        self.assertEqual(visuals[0].size, (2000, 1500))

        x1, y1, x2, y2 = module.union_box(entry["bbox"])
        self.assertEqual(visuals[1].size, (int(round(x2)) - int(x1), int(round(y2)) - int(y1)))

    def test_gtcrop_refuses_a_missing_box(self):
        """Falling back to one image would make this row a copy of the plain arm."""
        from PIL import Image

        module = load_generated(self.task_dir, "gtcrop")
        with self.assertRaises(KeyError):
            module.doc_to_visual({"image": Image.new("RGB", (64, 64)), "question_id": "no-such-id"})

    def test_gtcrop_prompt_adds_no_second_format_instruction(self):
        module = load_generated(self.task_dir, "gtcrop")
        text = module.doc_to_text({"text": "What colour?\n(A) red (B) blue"},
                                  {"pre_prompt": "", "post_prompt": "\nAnswer with the letter."})
        self.assertIn("second image", text)
        self.assertEqual(text.count("Answer with"), 1)


class Boxes(unittest.TestCase):
    """The committed V* boxes, checked against the benchmark they are keyed to."""

    @classmethod
    def setUpClass(cls):
        cls.boxes = json.loads((REPO / "assets" / "vstar_bench_boxes.json").read_text())

    def test_every_vstar_document_has_a_box(self):
        self.assertEqual(len(self.boxes), 191)
        self.assertEqual(sorted(int(k) for k in self.boxes), list(range(191)))

    def test_boxes_are_positive_and_in_xywh(self):
        for qid, entry in self.boxes.items():
            self.assertTrue(entry["bbox"], qid)
            for box in entry["bbox"]:
                self.assertEqual(len(box), 4, qid)
                x, y, w, h = box
                self.assertGreaterEqual(x, 0, qid)
                self.assertGreaterEqual(y, 0, qid)
                self.assertGreater(w, 0, qid)
                self.assertGreater(h, 0, qid)

    def test_the_boxes_are_the_needle_v_star_is_about(self):
        """If the median box were large, the oracle arm would not be an oracle."""
        areas = sorted(w * h for e in self.boxes.values() for _, _, w, h in e["bbox"])
        self.assertLess(areas[len(areas) // 2], 0.01 * 2000 * 1500)


class Report(unittest.TestCase):
    def test_verdict_thresholds(self):
        self.assertEqual(gate.verdict(12.0)[0], "PROCEED")
        self.assertEqual(gate.verdict(10.0)[0], "PROCEED")
        self.assertEqual(gate.verdict(5.0)[0], "PROCEED ONLY ON EFFICIENCY")
        self.assertEqual(gate.verdict(3.0)[0], "PROCEED ONLY ON EFFICIENCY")
        self.assertEqual(gate.verdict(2.9)[0], "STOP")
        self.assertEqual(gate.verdict(-4.0)[0], "STOP")
        self.assertEqual(gate.verdict(None)[0], "NO VERDICT")

    def _harvest(self, tmp):
        """Units in the scale lmms-eval really writes: V* 0-100, HR-Bench 0-1."""
        (tmp / "units").mkdir()
        for model, offsets in (("base", 0.0), ("coldstart", 5.0)):
            for cap, value in zip(gate.CAPS, (40.0, 52.0, 61.0)):
                (tmp / "units" / f"{model}_vstar_px{cap}.json").write_text(json.dumps(dict(
                    unit=f"{model}_vstar_px{cap}", model=model, cap=cap, tasks=["vstar_bench"],
                    metrics={"vstar_bench": value + offsets}, extras={}, minutes=3.0)))
            (tmp / "units" / f"{model}_vstarcrop_px640.json").write_text(json.dumps(dict(
                unit=f"{model}_vstarcrop_px640", model=model, cap=640,
                tasks=["vstar_bench_gtcrop"],
                metrics={"vstar_bench_gtcrop": 70.0 + offsets}, extras={}, minutes=3.0)))

    def test_the_two_scorers_are_on_different_scales(self):
        """V* harvests as a percentage and HR-Bench as a fraction.

        Scaling both by 100 is what made the first real report read +1832.5
        points on a +18.3-point gap -- and still print PROCEED, because the
        threshold is one-sided and a nonsense gap clears it.
        """
        tmp = Path(tempfile.mkdtemp())
        self._harvest(tmp)
        (tmp / "units" / "coldstart_hrbench_px640.json").write_text(json.dumps(dict(
            unit="coldstart_hrbench_px640", model="coldstart", cap=640,
            tasks=["hrbench4k_gate"], metrics={"hrbench4k_gate": 0.6205},
            extras={"hrbench4k_gate": {"parse_fail": 0.0}}, minutes=29.3)))
        data = gate.load_results(tmp)
        self.assertAlmostEqual(data[("coldstart", "vstar_bench", 640)]["value"], 45.0, places=4)
        self.assertAlmostEqual(data[("coldstart", "hrbench4k_gate", 640)]["value"], 62.05, places=4)

    def test_report_renders_from_harvested_units(self):
        tmp = Path(tempfile.mkdtemp())
        self._harvest(tmp)
        call, gap = gate.report(tmp, markdown=str(tmp / "out.md"))
        self.assertEqual(call, "PROCEED")
        self.assertAlmostEqual(gap, 21.0, places=5)
        text = (tmp / "out.md").read_text()
        # The oracle arm is a COLUMN on the V* row, not a row of its own: it is
        # the same 191 documents at the same cap, so putting it beside them is
        # what makes it readable as "and this is what the crop gives back".
        self.assertIn("640 + GT crop", text)
        self.assertIn("| coldstart | V* | 45.0 | 57.0 | 66.0 | 75.0 | +21.0 |", text)


class Cli(unittest.TestCase):
    def test_plan_runs_and_prints_every_unit(self):
        out = subprocess.run(
            [sys.executable, str(REPO / "ergo_gate_lowres.py"), "--plan",
             "--out-dir", tempfile.mkdtemp()],
            capture_output=True, text=True, check=True).stdout
        for unit in gate.plan():
            self.assertIn(unit["id"], out)
        self.assertIn("--force_simple", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
