# Redrob — Intelligent Candidate Discovery & Ranking

A hybrid, CPU-only ranking system for the **Data & AI Challenge** (Intelligent
Candidate Discovery & Ranking). It ranks the top-100 candidates from a
100,000-candidate pool against the released *Senior AI Engineer (Founding Team)*
job description — the way a great recruiter would, by **reading the profile**,
not by counting AI keywords.

> The dataset is deliberately adversarial: keyword stuffers (a "Marketing
> Manager" with every AI skill listed), plain-language strong candidates (who
> never say "RAG" but *built* the system), behavioral twins, and ~80 honeypots
> with subtly impossible profiles. The sample submission shipped with the bundle
> deliberately falls for the keyword trap. This system is built to beat it.

---

## TL;DR — how it works

Each candidate gets a single score from four interpretable layers (no LLM at
inference, so it scales and reproduces on CPU in minutes):

```
score = ( 0.70 · structured_fit  +  0.30 · semantic_fit )   # what & how-well they fit
        · behavioral_modifier                                # are they actually available?
        · honeypot_gate                                      # is the profile even possible?
```

1. **Structured fit** (the JD's explicit logic) — title coherence, real career
   evidence (built retrieval/ranking/recsys *in production*, with NDCG/MAP-style
   evaluation), experience band (6–8 yrs ideal), product-vs-services,
   NLP/IR-vs-CV/speech, location, and a **trust-weighted** skill score
   (endorsements × duration × proficiency × platform assessment) that ignores
   stuffed skill lists.
2. **Semantic fit** — a *contrastive* static-embedding signal:
   `cos(candidate, JD-ideal) − cos(candidate, JD-anti-pattern)`. This surfaces
   plain-language strong candidates and mutes stuffers (who look similar to
   both anchors). Runs on CPU with **no torch, no GPU** (model2vec static
   embeddings), with a pure-`scikit-learn` LSA fallback.
3. **Behavioral modifier** — down-weights the unavailable (dormant, low recruiter
   response, long notice), per the JD's explicit instruction. Calibrated to the
   observed signal distributions.
4. **Honeypot gate** — flags internal impossibilities (e.g. a current role
   claiming more tenure than has elapsed since it began; ≥3 "advanced/expert"
   skills with 0 months of use) and removes them from the rankable region.

Every ranked candidate ships with **fact-grounded reasoning** generated from the
candidate's own parsed fields (no hallucination), with honest concerns surfaced
and tone matched to the rank.

---

## Reproduce the submission

```bash
# 1. install (CPU-only; no torch)
pip install -r requirements.txt

# 2. one-time pre-computation: vendor the static embedding model locally
#    (the only step that uses the network; the ranking step is fully offline)
python scripts/fetch_model.py

# 3. produce the ranked top-100 CSV  (the timed ranking step)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`rank.py` also accepts:

| flag | default | meaning |
|------|---------|---------|
| `--embed {auto,model2vec,lsa}` | `auto` | semantic backend; `auto` uses the vendored model2vec if present, else LSA |
| `--ref-date YYYY-MM-DD` | `2026-06-21` | reference "today" for recency/date math (keeps runs reproducible) |
| `--top N` | `100` | how many to rank |
| `--limit N` | `0` | only read the first N candidates (debug) |

Validate before submitting (the official validator handles `.jsonl`/`.gz`):

```bash
python validate_submission.py submission.csv
```

---

## Compute & constraints

| Constraint | Limit | This system |
|------------|-------|-------------|
| Compute | CPU only, no GPU | ✅ static embeddings + numpy/sklearn, no torch |
| Network (ranking) | off | ✅ model vendored locally first; ranking makes no API calls |
| Memory | ≤ 16 GB | ✅ streams the pool; keeps only compact per-candidate scalars |
| Runtime (ranking) | ≤ 5 min | ✅ two-pass design; see `submission_metadata.yaml` |

The model download in step 2 is **pre-computation** (the spec allows this to
exceed the 5-minute window); the ranking step in step 3 is what must fit the
budget, and it does.

---

## Repository layout

```
rank.py                         # single-command entry point -> submission.csv
src/redrob_ranker/
  config.py                     # the JD, distilled: taxonomies, evidence patterns, weights
  loader.py                     # streaming JSONL(.gz) reader (orjson)
  textbuild.py                  # candidate -> embedding text & evidence blob
  embed.py                      # contrastive semantic layer (model2vec | LSA)
  structured.py                 # the JD-logic layer (title, evidence, skills, domain, ...)
  behavioral.py                 # availability modifier from the 23 redrob signals
  honeypot.py                   # internal-impossibility detection
  reasoning.py                  # deterministic, fact-grounded, varied reasoning
  scoring.py                    # blend + spec-valid ordering (ties -> candidate_id asc)
scripts/
  fetch_model.py                # one-time: vendor the static embedding model
  eda.py                        # exploratory analysis used to calibrate the ranker
  honeypot_diag.py              # honeypot-detector calibration
tests/                          # format + logic checks
artifacts/model/                # (gitignored) vendored model weights, via fetch_model.py
```

---

## Why this design

The JD's own closing note says the right answer is *"reasoning about the gap
between what the JD says and what the JD means."* So the system is built around
**reading evidence from career descriptions and the summary** (the truth-bearing
fields) and **distrusting the skills tag-cloud** (EDA shows skills are ~uniform
random noise — every common skill appears on ~12% of profiles, so "semantic
search" as a listed skill means almost nothing). Title coherence and demonstrated
production work do the heavy lifting; the contrastive embedding catches the
plain-language strong candidates; behavioral signals decide who is actually
hireable; and the honeypot gate keeps impossible profiles out of the top 100.

See `methodology` in `submission_metadata.yaml` and the accompanying deck for the
full write-up.

## AI tools

Built with AI assistance (declared in `submission_metadata.yaml`). No candidate
data is sent to any LLM at ranking time — the ranker is a deterministic,
inspectable scoring system, by design.
