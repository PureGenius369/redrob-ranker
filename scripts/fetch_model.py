"""
One-time build step: download a model2vec static embedding model and vendor it
into artifacts/model so the ranking step can run fully offline (no network).

    python scripts/fetch_model.py

Default model: minishlab/potion-retrieval-32M — a static embedding model
distilled specifically for retrieval, which is exactly our candidate<->JD
semantic-matching task. It runs on CPU at hashing speed (no transformer forward
pass, no GPU, no torch), so the whole ranking pipeline stays within budget.
"""
import os
import sys

MODEL = os.environ.get("REDROB_MODEL", "minishlab/potion-retrieval-32M")
FALLBACK = "minishlab/potion-base-8M"
DEST = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model")


def main():
    from model2vec import StaticModel

    dest = os.path.abspath(DEST)
    os.makedirs(dest, exist_ok=True)
    for name in (MODEL, FALLBACK):
        try:
            print(f"[fetch] downloading {name} ...")
            m = StaticModel.from_pretrained(name)
            m.save_pretrained(dest)
            dim = m.encode(["hello world"]).shape[1]
            print(f"[fetch] saved {name} -> {dest} (dim={dim})")
            return
        except Exception as e:
            print(f"[fetch] {name} failed: {e}")
    sys.exit("Could not fetch any model2vec model.")


if __name__ == "__main__":
    main()
