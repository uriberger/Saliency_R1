import os, time
os.environ.setdefault("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

t0 = time.time()
p = snapshot_download(
    "Xkev/LLaVA-CoT-100k", repo_type="dataset",
    local_dir="/home/uberger/scratch/research/saliency_r1/cold_data/LLaVA-CoT-100k",
    allow_patterns=["image.zip.part-*", "train.jsonl"],
    max_workers=8)
print("DONE", p, "in", round((time.time() - t0) / 60, 1), "min")
