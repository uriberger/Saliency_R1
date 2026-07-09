"""Smoke-test the Saliency-R1 generate path on the real Qwen3-VL-8B-Instruct (GPU).

Verifies that, with the ported SDPA identity trick applied (env saliency_r1_qwen3), a
`generate(output_attentions=True, ...)` call produces exactly the structures the patched
TRL trainer reads:
  - outputs.attentions        : [n_gen_steps][n_layers] tensors of [b, q_heads, q, kv]
  - outputs.hidden_states     : per-step, per-layer hidden states (deepstack-injected)
  - outputs.past_key_values.layers[l].values  (GQA kv cache)
and that the image-token count matches the trainer's reshape target (grid_h//2, grid_w//2).

Run on a GPU node:
  conda activate saliency_r1_qwen3
  HF_HOME=/home/uberger/scratch/cache/hf_cache python -u smoke_qwen3vl_generate_gpu.py
"""
import os, torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

os.environ.setdefault("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
assert torch.cuda.is_available(), "no CUDA device visible"
dev = "cuda"
torch.manual_seed(0)

print("loading processor + model (bf16, GPU)...", flush=True)
# moderate image cap keeps the identity trick's [b,h,seq,seq] attention matrices small
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=64*32*32, max_pixels=256*32*32)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
).to(dev).eval()

# confirm the patched op is actually loaded
import transformers.integrations.sdpa_attention as _sdpa, inspect
patched = "identity trick" in inspect.getsource(_sdpa.sdpa_attention_forward)
print("attn impl:", model.config._attn_implementation,
      "| SDPA identity-trick patch loaded:", patched,
      "| layers:", model.config.text_config.num_hidden_layers,
      "| q/kv heads:", model.config.text_config.num_attention_heads, "/",
      model.config.text_config.num_key_value_heads, flush=True)
assert patched, "the patched sdpa_attention.py is NOT active in this env"

img = Image.new("RGB", (224, 168), (120, 160, 200))
msgs = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": "What color is this image? Answer in one word."}]}]
inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt").to(dev)

grid = inputs["image_grid_thw"][0].tolist()
n_img_tok = int((inputs["input_ids"][0] == model.config.image_token_id).sum())
prompt_len = inputs["input_ids"].shape[1]
print(f"image_grid_thw={grid}  visual tokens expected t*(h/2)*(w/2)="
      f"{grid[0]*(grid[1]//2)*(grid[2]//2)}  actual(151655)={n_img_tok}  prompt_len={prompt_len}",
      flush=True)

print("generating 4 tokens with output_attentions + output_hidden_states...", flush=True)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=True,
                         return_dict_in_generate=True, output_attentions=True,
                         output_hidden_states=True)

gen = out.sequences[0, prompt_len:]
print("generated:", repr(proc.tokenizer.decode(gen, skip_special_tokens=True)), flush=True)

# ---- structural assertions ----
att = out.attentions
nL = model.config.text_config.num_hidden_layers
nH = model.config.text_config.num_attention_heads
nKV = model.config.text_config.num_key_value_heads
print("\n[attentions] n_gen_steps:", len(att), "| layers/step:", len(att[0]))
print("  prefill step[0] layer[0] shape:", tuple(att[0][0].shape), f"(expect [b,{nH},{prompt_len},{prompt_len}])")
print("  decode  step[1] layer[0] shape:", tuple(att[1][0].shape), f"(expect [b,{nH},1,{prompt_len+1}])")
rs = att[1][0][0, 0, -1].float().sum().item()
print("  decode last-query attn row-sum (expect ~1):", round(rs, 4))

hs = out.hidden_states
print("[hidden_states] n_gen_steps:", len(hs), "| (layers+1)/step:", len(hs[0]),
      "| last hs shape:", tuple(hs[0][-1].shape))

v0 = out.past_key_values.layers[0].values
print("[past_key_values] .layers[0].values shape:", tuple(v0.shape), f"(expect [b,{nKV},total_seq,head_dim])")

Hm, Wm = grid[1] // 2, grid[2] // 2
print(f"\n[reshape check] saliency map: {n_img_tok} img-token logits -> ({Hm},{Wm}) = {Hm*Wm}")

ok = (len(att[0]) == nL and att[0][0].shape[1] == nH
      and att[0][0].shape[-1] == prompt_len
      and abs(rs - 1.0) < 1e-2 and v0.shape[1] == nKV
      and n_img_tok == grid[0]*(grid[1]//2)*(grid[2]//2) == Hm*Wm)
print("\nSMOKE TEST PASSED:", bool(ok))
