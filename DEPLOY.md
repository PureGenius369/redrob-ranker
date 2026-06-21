# Deploying the sandbox demo

The submission requires a working hosted sandbox where the ranker can run on a
small candidate sample. Any one of the following satisfies it.

## Option A — Streamlit Community Cloud (easiest)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **New app** → pick this repo.
3. Set **Main file path** to `app.py`.
4. Add `streamlit` to the app's requirements (it's in `pyproject.toml`'s
   `sandbox` extra; on Streamlit Cloud add a line `streamlit` to a
   `requirements.txt` the platform reads, alongside the pinned deps).
5. Deploy. The app fetches the static model on first boot, then ranks an uploaded
   `.jsonl`/`.json` sample (or the bundled `sample_candidates.json`).

## Option B — HuggingFace Spaces (Streamlit SDK)

1. Create a new Space → SDK: **Streamlit** → CPU basic (free).
2. Push this repo's contents to the Space, and ensure the Space `README.md`
   starts with this frontmatter:

   ```yaml
   ---
   title: Redrob Candidate Ranker
   emoji: 🎯
   colorFrom: blue
   colorTo: indigo
   sdk: streamlit
   app_file: app.py
   pinned: false
   ---
   ```
3. Add `streamlit` to `requirements.txt` for the Space. It builds and serves
   `app.py` automatically.

## Option C — Docker (also serves as the Stage-3 reproduction recipe)

```bash
docker build -t redrob-ranker .
docker run --rm -v "$PWD":/data redrob-ranker \
  bash -lc "python scripts/precompute_embeddings.py --candidates /data/candidates.jsonl && \
            python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv"
```

## Option D — Google Colab

Open a notebook, then:

```python
!git clone https://github.com/<you>/redrob-ranker && cd redrob-ranker && pip install -r requirements.txt
!cd redrob-ranker && python scripts/fetch_model.py
# upload a small candidates sample, then:
!cd redrob-ranker && python rank.py --candidates sample.jsonl --out submission.csv --embed model2vec
```
