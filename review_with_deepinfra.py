#!/usr/bin/env python3
"""review_with_deepinfra.py — run the 3 ordered review angles on the audio-jepa code."""
import os, sys, json, urllib.request

KEY = os.environ["DEEPINFRA_API_KEY"]
URL = "https://api.deepinfra.com/v1/openai/chat/completions"

def call(model, system, user, max_tokens=4000):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}",
    })
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    # reasoning models put the final answer in `content` (possibly empty if they
    # only emitted reasoning); fall back to reasoning_content when needed.
    return (msg.get("content") or "") or (msg.get("reasoning_content") or "")

def main():
    repo = "/home/eileen/projects/fleet-jepa-midi"
    model_code = open(f"{repo}/audio_jepa/model.py").read()
    train_code = open(f"{repo}/train_audio_jepa.py").read()
    dataset_code = open(f"{repo}/audio_jepa/dataset.py").read()
    code_blob = f"=== model.py ===\n{model_code}\n\n=== dataset.py ===\n{dataset_code}\n\n=== train_audio_jepa.py ===\n{train_code}"

    jobs = [
        ("Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
         "You are a meticulous systems engineer reviewing PyTorch code for correctness bugs.",
         "Review this JEPA audio training code for BUGS and correctness issues. Focus on: "
         "shape mismatches, device placement, gradient flow through the EMA target / stop-gradient, "
         "VICReg implementation, masking correctness, the EMA momentum schedule, and any silent "
         "correctness bugs. Be specific with file/line-level findings and concrete fixes. Do not restate the code.\n\n" + code_blob),
        ("Qwen/Qwen3.6-35B-A3B",
         "You are a research mathematician. Verify the math of self-supervised JEPA objectives.",
         "Verify the mathematical correctness of this JEPA objective: EMA target encoder + stop-gradient + "
         "cosine-similarity predictor + VICReg variance/covariance anti-collapse. Specifically: (1) Is applying "
         "VICReg variance/covariance to L2-NORMALIZED embeddings (z=h/||h||) rather than raw h mathematically sound "
         "for preventing directional collapse, and is gamma=1/sqrt(D) the right hinge target? (2) Is the cosine-distance "
         "invariance term on normalized prediction vs stop-grad target equivalent to the repo's L1-on-sphere objective? "
         "(3) Any subtle issue with EMA momentum tau reaching/capping below 1.0 for anti-collapse? Answer concisely with math.\n\n" + code_blob),
        ("ByteDance/Seed-2.0-pro",
         "You are a senior ML architect critiquing a self-supervised audio representation design.",
         "Critique this architecture for a LEARNED audio encoder (replacing a hand-crafted vibe-matcher): mel frontend -> "
         "4 conv blocks -> 2-layer transformer -> mean-pool -> 384-dim, trained with JEPA (EMA target + stop-grad + cosine "
         "predictor + VICReg) on only 16 speech clips with mel-domain augmentations. What are the top risks (esp. collapse, "
         "overfitting on 16 clips), what would you change, and is this a reasonable first step? Be concrete and concise.\n\n" + code_blob),
    ]

    results = {}
    for model, sys_, user in jobs:
        try:
            out = call(model, sys_, user)
            results[model] = out
            print(f"\n{'='*70}\n### REVIEW: {model}\n{'='*70}\n{out}\n")
        except Exception as e:
            results[model] = f"ERROR: {e}"
            print(f"\n### {model}: ERROR {e}")

    with open("/tmp/ajepa_reviews.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
