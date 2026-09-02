"""What every existing set has spent from the Visual-CoT pools, and what is left for set_e.

Reuses build_set_d's accounting rather than re-deriving it: eightk_basenames() resolves
saliency-r1-8k both ways (metadata match + archive SHA-256 index), set_c_basenames()
replays set_c's seeded draw exactly, legacy_val_reference_keys() replays the
val_natural / val_nonnatural candidate pool, and draw_set_d() replays set_d.

Two exclusion policies, because they differ in whether set_e can reach vsr at all.

Run from the repo root:  PYTHONPATH=. MALLOC_ARENA_MAX=2 python outputs/hackset_analysis/pool_survey.py
"""
import os
import sys

sys.argv = ["survey"]
import build_grpo_sets as B
import build_set_c as C
import build_set_d as D

SCALE = 2  # set_e = SCALE x the 8k, in rows AND in images

meta = B.hf_snapshot(B.VISCOT_REPO, ["metadata/*.jsonl"], local_only=True) / "metadata"
cache = "cold_data/grpo_sets/_viscot_paths"

print("Loading Visual-CoT pools ...", flush=True)
pools = {s: C.load_pool(s, meta)[0] for s in D.RECIPE_ROWS}
groups_by_src = {s: C.group_by_image(pools[s]) for s in pools}

print("Resolving saliency-r1-8k ...", flush=True)
eightk, _, _ = D.eightk_basenames(meta, cache, pools)
print("Replaying set_c + the validation draws ...", flush=True)
set_c, val_c, legacy_val = D.set_c_basenames(meta)
print("Replaying set_a / set_b's Visual-CoT images ...", flush=True)
_, legacy_train = C.legacy_val_reference_keys(meta)
print("Replaying set_d ...", flush=True)
excl_d = eightk | set_c | val_c | set(legacy_val)
recs_d, _ = D.draw_set_d(meta, excl_d, pools)
set_d = {os.path.basename(r["_ref"][1]) for r in recs_d}
val_d = D.draw_val_d(meta, {r["_ref"][1] for r in recs_d}, excl_d, pools)
val_d_names = {os.path.basename(r["_ref"][1])
               for drawn, _ in val_d.values() for recs in drawn.values() for r in recs}

SETS = {
    "saliency-r1-8k": eightk,
    "set_a / set_b": set(legacy_train),
    "set_c": set_c,
    "set_d": set_d,
    "val_natural/nonnatural pool": set(legacy_val),
    "val_c pool": val_c,
    "val_d pool": val_d_names,
}
print("\nBasenames spoken for:")
for k, v in SETS.items():
    print(f"  {k:30s} {len(v):7d}")
print(f"  {'UNION':30s} {len(set().union(*SETS.values())):7d}")

VAL_POOL = set(legacy_val) | val_c | val_d_names
POLICIES = [
    ("STRICT - every set ever trained on or drawn for validation", list(SETS)),
    ("8k-LINEAGE - set_a / set_b left in, as build_set_c already allows",
     [k for k in SETS if k != "set_a / set_b"]),
]

for title, keys in POLICIES:
    spent = set().union(*[SETS[k] for k in keys])
    print(f"\n\n=== {title}  ->  {len(spent)} basenames excluded ===")
    print(f"{'source':16s}{'pool':>8}{'8k':>7}{'a/b':>7}{'set_c':>7}{'set_d':>7}"
          f"{'val*':>7}{'FREE':>8}{'need':>7}{'have':>7}{'+1q':>7}  verdict")
    ok, tot_free, tot_need = True, 0, 0
    for s in sorted(D.RECIPE_ROWS):
        g = groups_by_src[s]
        names = list(g)
        hit = lambda S: sum(1 for k in names if os.path.basename(k) in S)
        need_img, need_row = D.IMAGE_BUDGET[s] * SCALE, D.RECIPE_ROWS[s] * SCALE
        base, extra = divmod(need_row, need_img)
        free = {k: v for k, v in g.items() if os.path.basename(k) not in spent}
        lo = sum(1 for v in free.values() if len(v) >= base)
        hi = sum(1 for v in free.values() if len(v) >= base + 1)
        fits = lo >= need_img and hi >= extra
        ok &= fits
        tot_free += len(free)
        tot_need += need_img
        print(f"{s:16s}{len(g):8d}{hit(eightk):7d}{hit(SETS['set_a / set_b']):7d}"
              f"{hit(set_c):7d}{hit(set_d):7d}{hit(VAL_POOL):7d}{len(free):8d}"
              f"{need_img:7d}{lo:7d}{hi:7d}  {'OK' if fits else 'INFEASIBLE'}")
    print(f"{'TOTAL':16s}{'':>43}{tot_free:8d}{tot_need:7d}")
    print("FEASIBLE at 2x rows and 2x images" if ok else "INFEASIBLE - see the flagged row")
