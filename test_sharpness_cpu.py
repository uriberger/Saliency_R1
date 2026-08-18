#!/usr/bin/env python
"""CPU tests for the box-free sharpness columns and the report that scores them.

The scan they feed runs on 8 GPUs and nobody can iterate on it interactively, so
everything that can be checked without one is checked here: the seven statistics
against closed forms on maps whose answer is known by hand, the invariances that make
the comparison with the DINO metrics meaningful at all, the report's correlation
machinery against numpy, and -- the one that actually protects a conclusion -- the
calibration of the max-|r| permutation threshold under a true null.

    python test_sharpness_cpu.py [-v]

Real maps are also read from outputs/saliency_viz/*/samples/*/maps.npz when they are
present, which is the only place in this repo a saliency map is stored rather than
scored. Those tests skip themselves when the directory is not there.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SH = _load("_t_sharp", "saliency_sharpness.py")
SR = _load("_t_sharp_report", "sharpness_report.py")
NAMES = list(SH.SHARP_NAMES)
IX = {n: i for i, n in enumerate(NAMES)}


def one(maps, grid):
    """sharpness() on a single map -> a name->value dict."""
    s, neg = SH.sharpness(np.asarray(maps, dtype=np.float64)[None, :], grid)
    return {n: float(s[0, i]) for i, n in enumerate(NAMES)}, float(neg[0])


class TestClosedForms(unittest.TestCase):
    """Values a delta, a flat map and a two-point map must take, derived by hand."""

    def test_delta(self):
        for gh, gw in ((12, 16), (16, 16), (9, 16)):
            P = gh * gw
            x = np.zeros(P)
            x[P // 2] = 1.0
            v, _ = one(x, (gh, gw))
            self.assertAlmostEqual(v["cv"], np.sqrt(P - 1), places=6)
            self.assertAlmostEqual(v["nent"], 1.0, places=9)
            self.assertAlmostEqual(v["gini"], (P - 1) / P, places=9)
            self.assertAlmostEqual(v["top1"], 1.0, places=9)
            self.assertAlmostEqual(v["top5"], 1.0, places=9)
            self.assertAlmostEqual(v["top20"], 1.0, places=9)
            self.assertAlmostEqual(v["sconc"], 1.0, places=9)

    def test_flat(self):
        """A flat map must read exactly 0 on every column, at every grid shape.

        Not decoration: the grids in this corpus run 144-256 patches, so a statistic
        whose flat-map value moved with P would make image resolution look like
        sharpness and the whole correlation would be a readout of picture size.
        """
        for gh, gw in ((12, 16), (16, 10), (16, 16), (9, 16)):
            P = gh * gw
            v, _ = one(np.full(P, 3.7), (gh, gw))
            self.assertAlmostEqual(v["cv"], 0.0, places=9)
            self.assertAlmostEqual(v["nent"], 0.0, places=9)
            self.assertAlmostEqual(v["gini"], 0.0, places=9)
            self.assertAlmostEqual(v["top1"], 1.0 / P, places=9)
            self.assertAlmostEqual(v["top5"], np.ceil(0.05 * P) / P, places=9)
            self.assertAlmostEqual(v["top20"], np.ceil(0.20 * P) / P, places=9)
            self.assertAlmostEqual(v["sconc"], 0.0, places=9)

    def test_two_point(self):
        gh, gw = 12, 16
        P = gh * gw
        x = np.zeros(P)
        x[5 * gw + 3] = x[5 * gw + 4] = 0.5           # two adjacent cells, same row
        v, _ = one(x, (gh, gw))
        self.assertAlmostEqual(v["cv"], np.sqrt(P / 2 - 1), places=6)
        self.assertAlmostEqual(v["nent"], 1.0 - np.log(2) / np.log(P), places=9)
        self.assertAlmostEqual(v["gini"], (P - 2) / P, places=9)
        self.assertAlmostEqual(v["top1"], 0.5, places=9)
        var_flat = (1 - 1 / gh ** 2) / 12 + (1 - 1 / gw ** 2) / 12
        self.assertAlmostEqual(v["sconc"], 1 - np.sqrt(0.25 / gw ** 2 / var_flat),
                               places=9)

    def test_sconc_negative_for_two_distant_clumps(self):
        """Sharp in value, spread in space -- the case only `sconc` can see."""
        gh, gw = 12, 16
        x = np.zeros(gh * gw)
        x[0] = x[gh * gw - 1] = 0.5                   # opposite corners
        v, _ = one(x, (gh, gw))
        self.assertLess(v["sconc"], 0.0)
        self.assertGreater(v["cv"], 5.0)              # and the value columns call it sharp


class TestInvariances(unittest.TestCase):
    """The properties that make the comparison with the DINO metrics mean something."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.grid = (12, 16)
        self.x = rng.gamma(0.4, size=self.grid[0] * self.grid[1])

    def test_six_columns_are_blind_to_location(self):
        """Permuting the patches must not move them -- that is the whole argument.

        If a column moved under a permutation it could be encoding position, and
        "sharpness predicts correctness" would no longer be a claim that excludes
        grounding.
        """
        rng = np.random.default_rng(11)
        base, _ = one(self.x, self.grid)
        for _ in range(5):
            v, _ = one(rng.permutation(self.x), self.grid)
            for nm in SH.PERM_INVARIANT:
                self.assertAlmostEqual(v[nm], base[nm], places=9, msg=nm)

    def test_sconc_is_not(self):
        rng = np.random.default_rng(13)
        x = np.zeros(self.grid[0] * self.grid[1])
        x[:8] = 1.0                                   # one compact run of patches
        base, _ = one(x, self.grid)
        moved = [one(rng.permutation(x), self.grid)[0]["sconc"] for _ in range(5)]
        self.assertTrue(all(m < base["sconc"] - 0.05 for m in moved))

    def test_scale_invariance(self):
        base, _ = one(self.x, self.grid)
        for k in (1e-9, 1e3):
            v, _ = one(self.x * k, self.grid)
            for nm in NAMES:
                self.assertAlmostEqual(v[nm], base[nm], places=6, msg=nm)

    def test_monotone_in_temperature(self):
        """Sharpening a softmax must raise every column. Orientation check."""
        rng = np.random.default_rng(3)
        z = rng.normal(size=self.grid[0] * self.grid[1])
        prev = None
        for beta in (0.5, 1.0, 2.0, 4.0):
            e = np.exp(beta * (z - z.max()))
            v, _ = one(e, self.grid)
            if prev is not None:
                for nm in SH.PERM_INVARIANT:
                    self.assertGreater(v[nm], prev[nm], msg=f"{nm} at beta={beta}")
            prev = v


class TestDegenerate(unittest.TestCase):
    def test_negative_entries_are_rectified_and_reported(self):
        gh, gw = 12, 16
        x = np.full(gh * gw, -0.01)
        x[3] = 1.0
        v, neg = one(x, (gh, gw))
        pos_only = np.zeros(gh * gw)
        pos_only[3] = 1.0
        ref, _ = one(pos_only, (gh, gw))
        for nm in NAMES:
            self.assertAlmostEqual(v[nm], ref[nm], places=9, msg=nm)
        self.assertAlmostEqual(neg, (gh * gw - 1) * 0.01 / (1.0 + (gh * gw - 1) * 0.01),
                               places=9)

    def test_all_zero_map_is_nan_not_zero(self):
        v, neg = one(np.zeros(192), (12, 16))
        for nm in NAMES:
            self.assertTrue(np.isnan(v[nm]), msg=nm)
        self.assertTrue(np.isnan(neg))

    def test_shape_and_grid_mismatch_raises(self):
        with self.assertRaises(ValueError):
            SH.sharpness(np.ones((2, 100)), (12, 16))

    def test_leading_axes_are_preserved(self):
        rng = np.random.default_rng(5)
        maps = rng.gamma(0.5, size=(36, 32, 3, 192))
        s, neg = SH.sharpness(maps, (12, 16))
        self.assertEqual(s.shape, (36, 32, 3, len(NAMES)))
        self.assertEqual(neg.shape, (36, 32, 3))
        # and a single map scored alone must match its cell in the batch
        v, _ = one(maps[7, 11, 2], (12, 16))
        for i, nm in enumerate(NAMES):
            self.assertAlmostEqual(s[7, 11, 2, i], v[nm], places=9, msg=nm)


class TestScanPlumbing(unittest.TestCase):
    """The slicing the two scans do around sharpness(), without needing their GPUs.

    Both probes accumulate one row per STEP out of an array whose step axis sits in a
    different place -- [L,H,S,M] in the head scan, [K,1,S,M] in the flow scan -- and an
    index slip there would silently mis-align sharpness with the correctness label it
    is correlated against. Nothing downstream could detect that, so it is pinned here.
    """

    def test_head_scan_layout(self):
        rng = np.random.default_rng(31)
        L, H, S, P = 36, 32, 3, 192
        maps = rng.gamma(0.5, size=(L, H, S, P))
        sh, neg = SH.sharpness(maps, (12, 16))
        mass = maps.sum(-1)
        rows = [sh[:, :, si] for si in range(S)]
        negs = [neg[:, :, si] for si in range(S)]
        masses = [mass[:, :, si] for si in range(S)]
        self.assertEqual(np.stack(rows).shape, (S, L, H, len(NAMES)))
        self.assertEqual(np.stack(negs).shape, (S, L, H))
        self.assertEqual(np.stack(masses).shape, (S, L, H))
        # step 2's row must be step 2's map, not step 0's
        v, _ = one(maps[5, 9, 2], (12, 16))
        for i, nm in enumerate(NAMES):
            self.assertAlmostEqual(rows[2][5, 9, i], v[nm], places=9, msg=nm)

    def test_flow_scan_layout(self):
        rng = np.random.default_rng(32)
        K, S, P = 4, 3, 192
        maps = rng.gamma(0.5, size=(K, 1, S, P))
        sh, neg = SH.sharpness(maps, (12, 16))
        rows = [sh[:, 0, si] for si in range(S)]
        self.assertEqual(np.stack(rows).shape, (S, K, len(NAMES)))
        self.assertEqual(np.stack([neg[:, 0, si] for si in range(S)]).shape, (S, K))
        v, _ = one(maps[2, 0, 1], (12, 16))
        for i, nm in enumerate(NAMES):
            self.assertAlmostEqual(rows[1][2, i], v[nm], places=9, msg=nm)


class TestRealMaps(unittest.TestCase):
    """The stored maps from saliency_viz: five methods, real images, real steps."""

    @classmethod
    def setUpClass(cls):
        cls.files = sorted(glob.glob(
            str(REPO / "outputs/saliency_viz/*/samples/*/maps.npz")))

    def test_every_method_scores_finite_and_in_range(self):
        if not self.files:
            self.skipTest("no outputs/saliency_viz samples on this machine")
        seen = set()
        for f in self.files[:20]:
            z = np.load(f)
            meta = Path(f).parent / "meta.json"
            import json
            grid = tuple(json.loads(meta.read_text())["grid"])
            for k in z.files:
                arr = z[k]
                if arr.ndim != 3:
                    continue
                s, _ = SH.sharpness(arr.reshape(arr.shape[0], -1), grid)
                seen.add(k)
                self.assertTrue(np.isfinite(s).all(), f"{k} in {f}")
                for nm in ("nent", "gini", "top1", "top5", "top20"):
                    v = s[:, IX[nm]]
                    self.assertTrue(((v >= -1e-9) & (v <= 1 + 1e-9)).all(),
                                    f"{nm} out of [0,1] for {k} in {f}")
                self.assertTrue((s[:, IX["cv"]] >= -1e-9).all(), k)
                self.assertTrue((s[:, IX["sconc"]] <= 1 + 1e-9).all(), k)
        self.assertGreaterEqual(len(seen), 4, f"only saw {seen}")

    def test_real_maps_sit_between_flat_and_delta(self):
        if not self.files:
            self.skipTest("no outputs/saliency_viz samples on this machine")
        import json
        z = np.load(self.files[0])
        grid = tuple(json.loads((Path(self.files[0]).parent / "meta.json")
                                .read_text())["grid"])
        P = grid[0] * grid[1]
        for k in z.files:
            arr = z[k]
            if arr.ndim != 3:
                continue
            s, _ = SH.sharpness(arr.reshape(arr.shape[0], -1), grid)
            self.assertTrue((s[:, IX["cv"]] < np.sqrt(P - 1)).all(), k)
            self.assertTrue((s[:, IX["nent"]] < 1.0).all(), k)


class TestReportStats(unittest.TestCase):
    def test_col_corr_matches_numpy(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(300, 9))
        y = X[:, 0] * 0.4 + rng.normal(size=300)
        got = SR.col_corr(X, y)
        want = np.array([np.corrcoef(X[:, k], y)[0, 1] for k in range(9)])
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_col_corr_is_nan_aware(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(200, 3))
        y = rng.normal(size=200)
        X[::7, 1] = np.nan
        got = SR.col_corr(X, y)
        ok = np.isfinite(X[:, 1])
        self.assertAlmostEqual(got[1], np.corrcoef(X[ok, 1], y[ok])[0, 1], places=12)

    def test_partial_removes_a_planted_confound(self):
        """y and X share only a common cause: the raw r is large, the partial is not."""
        rng = np.random.default_rng(6)
        z = rng.normal(size=500)
        X = np.column_stack([z + 0.3 * rng.normal(size=500) for _ in range(4)])
        y = z + 0.3 * rng.normal(size=500)
        Z = z[:, None]
        self.assertGreater(abs(SR.col_corr(X, y)[0]), 0.8)
        self.assertLess(abs(SR.partial_col_corr(X, y, Z)[0]), 0.15)

    def test_pairwise_partial_uses_each_column_own_partner(self):
        rng = np.random.default_rng(8)
        c0, c1 = rng.normal(size=500), rng.normal(size=500)
        y = c0 + 0.2 * rng.normal(size=500)
        X = np.column_stack([c0 + 0.2 * rng.normal(size=500), rng.normal(size=500)])
        got = SR.pairwise_partial(X, y, np.column_stack([c0, c1]))
        self.assertLess(abs(got[0]), 0.3)      # its own partner explains it away
        self.assertLess(abs(got[1]), 0.2)      # and column 1 was noise to begin with

    def test_max_abs_threshold_is_calibrated(self):
        """Under a true null the observed max |r| must clear thr95 about 5% of the time.

        This is the number every "* clears its threshold" in the report rests on, and
        it has to hold for CORRELATED columns -- 1,152 neighbouring attention heads are
        nothing like 1,152 independent tests, which is exactly why Bonferroni is not
        used for the headline.
        """
        rng = np.random.default_rng(21)
        n, k = 300, 40
        common = rng.normal(size=(n, 1))
        X = common + 0.6 * rng.normal(size=(n, k))       # heavily correlated columns
        y0 = (rng.random(n) < 0.45).astype(float)
        thr, null, nimp = SR.max_abs_null(X, y0, 800, seed=1)
        self.assertEqual(nimp, 0)
        exceed = 0
        trials = 300
        for t in range(trials):
            y = rng.permutation(y0)
            exceed += abs(SR.col_corr(X, y)).max() >= thr
        rate = exceed / trials
        self.assertGreater(rate, 0.01, f"threshold too loose: {rate:.3f}")
        self.assertLess(rate, 0.12, f"threshold too tight: {rate:.3f}")

    def test_max_abs_threshold_grows_with_the_number_of_tests(self):
        rng = np.random.default_rng(22)
        n = 300
        y = (rng.random(n) < 0.5).astype(float)
        small = SR.max_abs_null(rng.normal(size=(n, 5)), y, 400, seed=2)[0]
        big = SR.max_abs_null(rng.normal(size=(n, 400)), y, 400, seed=2)[0]
        self.assertGreater(big, small * 1.2)


# ---------------------------------------------------------------------------
# end-to-end: fabricate the two scan layouts and run the whole report
# ---------------------------------------------------------------------------
def fake_scan(dirpath, kind, n_cases=60, seed=0, planted=0.0):
    """Write shard npz files in the layout a real scan produces.

    `planted` correlates one sharpness column with correctness so the report has
    something to find, which is how "it ran" is distinguished from "it worked".
    """
    rng = np.random.default_rng(seed)
    out = Path(dirpath) / "scan"
    out.mkdir(parents=True, exist_ok=True)
    for shard in range(2):
        rows, steps, cor, uni, npat, ntok, ds = [], [], [], [], [], [], []
        V2, AU, MASS, SHV, NEG = [], [], [], [], []
        for c in range(n_cases // 2):
            row = shard + 2 * c
            grade = float(rng.random() < 0.5)
            ns = int(rng.integers(1, 4))
            gh, gw = (12, 16) if row % 3 else (10, 16)
            for si in range(ns):
                rows.append(row)
                steps.append(si)
                cor.append(grade)
                uni.append(float(rng.random() * 0.9 + 0.05))
                npat.append(gh * gw)
                ntok.append(int(rng.integers(6, 40)))
                ds.append("gqa" if row % 2 else "aokvqa")
                shape = (36, 32) if kind == "head" else (4,)
                V2.append(rng.normal(1.0, 0.2, size=shape).astype(np.float32))
                AU.append(rng.normal(0.5, 0.05, size=shape).astype(np.float32))
                MASS.append(rng.normal(0.3, 0.05, size=shape).astype(np.float32))
                s = rng.normal(0.5, 0.1, size=shape + (len(NAMES),))
                s[..., 0] += planted * grade
                SHV.append(s.astype(np.float32))
                NEG.append(np.zeros(shape, dtype=np.float32))
        kw = {}
        if kind == "head":
            kw["layers"] = np.arange(36)
        else:
            kw["names"] = np.array(["gnorm", "gnorm_ds", "gxi", "gxi_ds"])
            kw["map"] = np.array("grad")
            kw["alpha"] = np.array(0.5)
        np.savez_compressed(
            out / f"shard{shard:02d}.npz",
            v2=np.stack(V2), auroc=np.stack(AU), mass=np.stack(MASS),
            sharp=np.stack(SHV), neg_frac=np.stack(NEG),
            sharp_names=np.array(NAMES),
            row=np.array(rows), step=np.array(steps),
            correct=np.array(cor, dtype=np.float32),
            union=np.array(uni, dtype=np.float32),
            npatch=np.array(npat), ntok=np.array(ntok), dataset=np.array(ds), **kw)
    return dirpath


class TestEndToEnd(unittest.TestCase):
    def test_report_runs_on_both_scan_layouts(self):
        with tempfile.TemporaryDirectory() as td:
            fake_scan(Path(td) / "heads", "head", seed=1, planted=0.0)
            fake_scan(Path(td) / "grad", "flow", seed=2, planted=0.6)
            argv = sys.argv
            sys.argv = ["sharpness_report.py",
                        "--scan", f"heads={Path(td) / 'heads'}",
                        "--scan", f"grad={Path(td) / 'grad'}",
                        "--perm", "200", "--json", str(Path(td) / "out.json")]
            try:
                import io
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    SR.main()
            finally:
                sys.argv = argv
            text = buf.getvalue()
            self.assertIn("HEADLINE", text)
            self.assertIn("mean all heads", text)
            self.assertIn("gnorm", text)
            import json
            js = json.loads((Path(td) / "out.json").read_text())
            self.assertEqual(len(js), 2)
            # the planted column must be found in `grad`, and it must be `cv` (index 0)
            grad = [f for f in js if f["name"] == "grad"][0]
            self.assertIn("cv", grad["best"]["SHARP"]["label"])
            self.assertGreater(grad["best"]["SHARP"]["r_all"], 0.2)
            self.assertLess(grad["best"]["SHARP"]["p_fw"], 0.05)
            # and the pure-noise head family must NOT clear its own threshold
            heads = [f for f in js if f["name"] == "heads"][0]
            self.assertGreaterEqual(heads["best"]["SHARP"]["p_fw"], 0.05)

    def test_missing_sharpness_columns_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "old"
            (d / "scan").mkdir(parents=True)
            np.savez_compressed(d / "scan" / "shard00.npz",
                                v2=np.ones((4, 2)), auroc=np.ones((4, 2)),
                                row=np.arange(4), step=np.zeros(4),
                                correct=np.zeros(4, dtype=np.float32),
                                union=np.full(4, 0.3, dtype=np.float32),
                                names=np.array(["a", "b"]))
            with self.assertRaises(SystemExit) as cm:
                SR.Family("old", d)
            self.assertIn("predates", str(cm.exception))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    unittest.main(verbosity=2)
