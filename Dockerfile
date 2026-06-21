# Reproducible, CPU-only environment for the Redrob ranker.
# Build:  docker build -t redrob-ranker .
# Run (mount the candidate pool as /data):
#   docker run --rm -v "$PWD":/data redrob-ranker \
#     bash -lc "python scripts/precompute_embeddings.py --candidates /data/candidates.jsonl && \
#               python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv"
#
# The model is vendored at build time (network allowed during build). The ranking
# step (rank.py) runs fully offline. If precomputed embeddings are absent, rank.py
# transparently falls back to encoding inline, so `docker run ... python rank.py`
# also works out of the box.

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Pre-computation: vendor the static embedding model into artifacts/model.
RUN HF_HUB_OFFLINE=0 python scripts/fetch_model.py

CMD ["python", "rank.py", "--candidates", "/data/candidates.jsonl", "--out", "/data/submission.csv"]
