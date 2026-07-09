import os
os.environ.setdefault("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download
p = snapshot_download(
    "HuanjinYao/Mulberry-SFT", repo_type="dataset",
    local_dir="/home/uberger/scratch/research/saliency_r1/cold_data/Mulberry-SFT",
    allow_patterns=["mulberry_images.tar"], max_workers=8)
print("DONE", p)
