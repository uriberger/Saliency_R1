import os, openai

base_url = os.environ.get("OPENAI_BASE_URL", "https://inference-api.nvidia.com/v1")
key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
print("base_url      :", base_url)
print("key source    :", "OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY") else "NVIDIA_API_KEY")
print("key (masked)  :", (key[:6] + "..." + key[-4:]) if key else "(MISSING)")
print("JUDGE_MODEL   :", os.environ.get("JUDGE_MODEL", "(unset)"))

client = openai.OpenAI(api_key=key, base_url=base_url)

print("\n--- models this key can list ---")
try:
    for m in client.models.list().data:
        print("  ", m.id)
except Exception as e:
    print("  models.list() failed:", repr(e))

for model in ["gpt-4o-mini", "azure/openai/gpt-4o-mini"]:
    print(f"\n--- chat.completions with model={model!r} ---")
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=8,
            messages=[{"role": "user", "content": "reply with the word ok"}],
        )
        print("  OK:", r.choices[0].message.content)
    except Exception as e:
        print("  FAILED:", repr(e))
