"""
Build the approach deck as a PDF (landscape slides) with reportlab — no external
converter needed. Reads the final submission CSV + candidate pool to embed real
top-100 statistics.

    python scripts/build_deck.py --candidates ./candidates.jsonl \
        --submission ./outputs/submission_full.csv --out ./outputs/Redrob_Approach_Deck.pdf
"""
from __future__ import annotations
import argparse, csv, os, sys, datetime as dt
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from redrob_ranker import loader, honeypot, config  # noqa: E402

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

NAVY = HexColor("#0F2742")
ACCENT = HexColor("#2E86DE")
TEAL = HexColor("#0E9F6E")
RED = HexColor("#E64A40")
LIGHT = HexColor("#EEF2F6")
MUTED = HexColor("#5B6B7B")
DARK = HexColor("#16202B")
WHITE = HexColor("#FFFFFF")

W, H = landscape(A4)  # 842 x 595


def stats(candidates_path, submission_path):
    rows = []
    with open(submission_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    top_ids = {row["candidate_id"] for row in rows}
    recs = {}
    for c in loader.iter_candidates(candidates_path):
        if c["candidate_id"] in top_ids:
            recs[c["candidate_id"]] = c
            if len(recs) == len(top_ids):
                break
    ref = dt.date.fromisoformat(config.REFERENCE_DATE)
    titles = Counter()
    india = otw = hp = 0
    yoe_sum = 0.0
    for cid, c in recs.items():
        p = c.get("profile", {})
        titles[p.get("current_title", "?")] += 1
        if "india" in (p.get("country", "").lower()):
            india += 1
        if (c.get("redrob_signals") or {}).get("open_to_work_flag"):
            otw += 1
        yoe_sum += p.get("years_of_experience") or 0
        if honeypot.detect(c, ref)["is_honeypot"]:
            hp += 1
    n = len(recs)
    return {
        "n": n,
        "titles": titles.most_common(8),
        "india_pct": round(100 * india / n),
        "otw_pct": round(100 * otw / n),
        "mean_yoe": round(yoe_sum / n, 1),
        "honeypots": hp,
        "top5": rows[:5],
        "score_hi": rows[0]["score"],
        "score_lo": rows[-1]["score"],
    }


class Deck:
    def __init__(self, path):
        self.c = canvas.Canvas(path, pagesize=landscape(A4))
        self.page = 0

    def _footer(self):
        self.c.setFont("Helvetica", 7)
        self.c.setFillColor(MUTED)
        self.c.drawString(18 * mm, 8 * mm, "Redrob — Intelligent Candidate Discovery & Ranking")
        self.c.drawRightString(W - 18 * mm, 8 * mm, f"{self.page}")

    def header(self, kicker, title, color=NAVY):
        self.page += 1
        self.c.setFillColor(WHITE)
        self.c.rect(0, 0, W, H, fill=1, stroke=0)
        self.c.setFillColor(color)
        self.c.rect(0, H - 30 * mm, W, 30 * mm, fill=1, stroke=0)
        self.c.setFillColor(ACCENT)
        self.c.rect(0, H - 30 * mm, W, 2.2 * mm, fill=1, stroke=0)
        self.c.setFont("Helvetica-Bold", 10)
        self.c.setFillColor(ACCENT)
        self.c.drawString(18 * mm, H - 13 * mm, kicker.upper())
        self.c.setFont("Helvetica-Bold", 21)
        self.c.setFillColor(WHITE)
        self.c.drawString(18 * mm, H - 24 * mm, title)
        self._footer()

    def bullets(self, items, x=20 * mm, y=H - 45 * mm, dy=11 * mm, size=12, gap=6):
        for it in items:
            if isinstance(it, tuple):
                head, sub = it
            else:
                head, sub = it, None
            self.c.setFillColor(ACCENT)
            self.c.circle(x + 1.5 * mm, y + 1.3 * mm, 1.3 * mm, fill=1, stroke=0)
            self.c.setFont("Helvetica-Bold", size)
            self.c.setFillColor(DARK)
            self.c.drawString(x + 6 * mm, y, head)
            if sub:
                self.c.setFont("Helvetica", size - 2)
                self.c.setFillColor(MUTED)
                self.c.drawString(x + 6 * mm, y - 5.2 * mm, sub)
                y -= dy + gap * mm
            else:
                y -= dy
        return y

    def box(self, x, y, w, h, fill, text, sub=None, tcolor=WHITE):
        self.c.setFillColor(fill)
        self.c.roundRect(x, y, w, h, 3 * mm, fill=1, stroke=0)
        self.c.setFillColor(tcolor)
        self.c.setFont("Helvetica-Bold", 11)
        self.c.drawCentredString(x + w / 2, y + h - 8.5 * mm, text)
        if sub:
            self.c.setFont("Helvetica", 8)
            ty = y + h - 14 * mm
            for line in sub:
                self.c.drawCentredString(x + w / 2, ty, line)
                ty -= 4.2 * mm

    def save(self):
        self.c.showPage()
        self.c.save()


def build(st, out, team, date):
    d = Deck(out)
    c = d.c

    # ---- Slide 1: title ----
    d.page += 1
    c.setFillColor(NAVY); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(ACCENT); c.rect(0, H - 4 * mm, W, 4 * mm, fill=1, stroke=0)
    c.setFillColor(ACCENT); c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, H - 38 * mm, "DATA & AI CHALLENGE  ·  INTELLIGENT CANDIDATE DISCOVERY & RANKING")
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold", 38)
    c.drawString(20 * mm, H - 72 * mm, "Reading the profile,")
    c.drawString(20 * mm, H - 88 * mm, "not the keywords.")
    c.setFillColor(HexColor("#9FB3C8")); c.setFont("Helvetica", 14)
    c.drawString(20 * mm, H - 104 * mm, "A hybrid, CPU-only ranker that beats keyword-stuffers,")
    c.drawString(20 * mm, H - 112 * mm, "surfaces plain-language talent, and avoids the honeypots.")
    c.setStrokeColor(ACCENT); c.setLineWidth(2); c.line(20 * mm, H - 120 * mm, 60 * mm, H - 120 * mm)
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 30 * mm, team)
    c.setFillColor(MUTED); c.setFont("Helvetica", 10)
    c.drawString(20 * mm, 23 * mm, f"Senior AI Engineer (Founding Team) JD   ·   {date}")
    c.showPage()

    # ---- Slide 2: the problem / the trap ----
    d.header("The problem", "Keyword filters can't see what matters")
    d.bullets([
        ("Recruiters miss the right person — not for lack of talent, but because keyword filters are blind to fit.",
         "The JD is written to defeat keyword matching; the 'right answer' is reasoning about what the JD means."),
        ("The dataset is adversarial by design.",
         "Keyword stuffers, plain-language strong candidates, behavioral twins, and ~80 'impossible' honeypots."),
        ("The shipped sample submission deliberately falls for it.",
         "It ranks an HR Manager #1 and real ML Engineers at #27 / #48 / #99 — purely on AI-skill counts."),
        ("Our goal: a shortlist a recruiter can trust — and that survives code reproduction + interview.",
         "NDCG@10 dominates scoring (0.50), so the top-10 must be genuinely excellent."),
    ])
    c.showPage()

    # ---- Slide 3: what the JD means ----
    d.header("Reading between the lines", "What the role actually needs")
    c.setFillColor(TEAL); c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, H - 42 * mm, "FIT  (the ideal hire)")
    d.bullets([
        ("6–8 yrs, applied ML at product companies (not services)", None),
        ("Shipped search / ranking / retrieval / recsys to real users", None),
        ("Embeddings + vector/hybrid search + rigorous ranking eval", None),
        ("Strong NLP/IR; in/near Pune–Noida; actually available", None),
    ], x=20 * mm, y=H - 52 * mm, dy=10 * mm, size=11, gap=0)
    c.setFillColor(RED); c.setFont("Helvetica-Bold", 12)
    c.drawString(120 * mm, H - 42 * mm, "ANTI-FIT  (explicit disqualifiers)")
    d.bullets([
        ("Stuffed AI skills but a non-technical real title", None),
        ("Pure research / academia with no production", None),
        ("Only recent LangChain-on-OpenAI wrappers", None),
        ("CV/speech-only · services-only · title-chasers · dormant", None),
    ], x=120 * mm, y=H - 52 * mm, dy=10 * mm, size=11, gap=0)
    c.showPage()

    # ---- Slide 4: architecture ----
    d.header("Architecture", "Four interpretable layers, no LLM at inference")
    c.setFillColor(LIGHT); c.roundRect(20 * mm, H - 70 * mm, W - 40 * mm, 18 * mm, 3 * mm, fill=1, stroke=0)
    c.setFillColor(DARK); c.setFont("Courier-Bold", 12)
    c.drawCentredString(W / 2, H - 60 * mm, "score = (0.70·structured_fit + 0.30·semantic_fit) · behavioral_modifier · honeypot_gate")
    bw, bh, by = 38 * mm, 34 * mm, H - 120 * mm
    xs = [20 * mm, 70 * mm, 120 * mm, 170 * mm]
    d.box(xs[0], by, bw, bh, NAVY, "Structured fit", ["title coherence", "career evidence", "skill trust", "domain · location"])
    d.box(xs[1], by, bw, bh, ACCENT, "Semantic fit", ["contrastive", "JD-ideal minus", "JD-anti-pattern", "static embeds"])
    d.box(xs[2], by, bw, bh, TEAL, "Behavioral", ["availability", "recency · response", "open-to-work", "notice period"])
    d.box(xs[3], by, bw, bh, RED, "Honeypot gate", ["impossible", "tenure / skills", "→ removed from", "top-100"])
    c.setFillColor(MUTED); c.setFont("Helvetica-Oblique", 10)
    c.drawString(20 * mm, by - 9 * mm, "Two-pass design: scalar scoring over 100K, then full facts + grounded reasoning for the top 100. CPU-only, streams the pool.")
    c.showPage()

    # ---- Slide 5: structured ----
    d.header("Layer 1 — Structured fit", "The JD's logic, made machine-usable (70% weight)")
    d.bullets([
        ("Title coherence — the decisive anti-stuffer signal.",
         "A 'Marketing Manager' with 9 AI skills scores ~0.06; bullseye AI/search/recsys titles score 1.0."),
        ("Career evidence — read from descriptions, not skill tags.",
         "Credits demonstrated retrieval / ranking / production / NDCG-MAP evaluation / NLP-IR work."),
        ("Skill trust — defeats stuffing.",
         "Skills reweighted by endorsements × duration × proficiency × platform assessment (raw lists ignored)."),
        ("Plus: experience band (6–8), product-vs-services, NLP/IR-vs-CV/speech, location; disqualifier penalties.",
         "Research-only, LangChain-only, and title-chaser flags apply explicit multiplicative penalties."),
    ])
    c.showPage()

    # ---- Slide 6: semantic ----
    d.header("Layer 2 — Semantic fit", "Contrastive static embeddings (30% weight, CPU, no GPU)")
    d.bullets([
        ("sem = cos(candidate, JD-ideal)  −  cos(candidate, JD-anti-pattern).",
         "Subtracting similarity to the unwanted profile is what mutes stuffers (close to both anchors)."),
        ("Surfaces plain-language Tier-5s.",
         "Someone titled 'Software Engineer' who describes building a recommendation system rises on semantics."),
        ("model2vec static embeddings — distilled for retrieval.",
         "No transformer forward pass, no torch, no GPU; the latency/quality tradeoff the JD explicitly asks about."),
        ("Pure scikit-learn LSA fallback keeps it reproducible with zero downloads.", None),
    ])
    c.showPage()

    # ---- Slide 7: behavioral + honeypot ----
    d.header("Layers 3 & 4 — Available & possible", "Down-weight the unavailable; remove the impossible")
    c.setFillColor(TEAL); c.setFont("Helvetica-Bold", 12); c.drawString(20 * mm, H - 42 * mm, "Behavioral modifier")
    d.bullets([
        ("'Dormant 6 months + 5% response = not actually available.'", None),
        ("Bounded multiplier from 23 signals, calibrated to the pool.", None),
        ("Recency, response rate, open-to-work, notice, demand.", None),
    ], x=20 * mm, y=H - 52 * mm, dy=10 * mm, size=11, gap=0)
    c.setFillColor(RED); c.setFont("Helvetica-Bold", 12); c.drawString(120 * mm, H - 42 * mm, "Honeypot gate")
    d.bullets([
        ("Detects internal impossibilities, not specific IDs.", None),
        ("Tenure > time elapsed; ≥3 expert skills at 0 months.", None),
        (f"{st['honeypots']} honeypots in our top-100 (cap is 10%).", None),
    ], x=120 * mm, y=H - 52 * mm, dy=10 * mm, size=11, gap=0)
    c.showPage()

    # ---- Slide 8: EDA insights ----
    d.header("What the data told us", "EDA on the full 100K pool drove every threshold")
    d.bullets([
        ("Skills are ~uniform random noise.",
         "Every common skill sits at ~12% of profiles — 'semantic search' as a listed skill means almost nothing."),
        ("Only ~1,200 of 100,000 have a genuine AI/ML title.",
         "~68K carry the 12 non-technical 'stuffer' titles; the bullseye titles number in the dozens."),
        ("Honeypots are rare and detectable.",
         "We gate 43 internally-impossible profiles pool-wide via tenure/skill contradictions."),
        ("Availability is a gradient, not a cliff.",
         "Everyone is ≥45 days inactive (median 130) — recency is scaled to the real distribution."),
    ])
    c.showPage()

    # ---- Slide 9: results ----
    d.header("Results", "A top-100 a recruiter can trust")
    cards = [
        (f"{st['india_pct']}%", "based in India"),
        (f"{st['otw_pct']}%", "open to work"),
        (f"{st['mean_yoe']}", "mean years exp."),
        (f"{st['honeypots']}", "honeypots in top-100"),
    ]
    cx = 20 * mm
    for big, lab in cards:
        c.setFillColor(LIGHT); c.roundRect(cx, H - 70 * mm, 44 * mm, 22 * mm, 3 * mm, fill=1, stroke=0)
        c.setFillColor(ACCENT); c.setFont("Helvetica-Bold", 22); c.drawString(cx + 5 * mm, H - 60 * mm, big)
        c.setFillColor(MUTED); c.setFont("Helvetica", 9); c.drawString(cx + 5 * mm, H - 66 * mm, lab)
        cx += 49 * mm
    c.setFillColor(DARK); c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, H - 80 * mm, "Sample top-ranked reasoning (generated from the candidate's own facts, no hallucination):")
    y = H - 88 * mm
    for row in st["top5"][:4]:
        c.setFillColor(ACCENT); c.setFont("Helvetica-Bold", 9)
        c.drawString(20 * mm, y, f"#{row['rank']}")
        c.setFillColor(DARK); c.setFont("Helvetica", 8.3)
        text = row["reasoning"]
        # wrap
        words, line, lines = text.split(), "", []
        for w_ in words:
            if c.stringWidth(line + " " + w_, "Helvetica", 8.3) < (W - 52 * mm):
                line = (line + " " + w_).strip()
            else:
                lines.append(line); line = w_
        lines.append(line)
        for ln in lines[:2]:
            c.drawString(30 * mm, y, ln); y -= 4.4 * mm
        y -= 2.5 * mm
    c.showPage()

    # ---- Slide 10: reproducibility ----
    d.header("Reproducible & defensible", "Built for Stage-3 reproduction and the interview")
    d.bullets([
        ("One command, CPU-only, no network at ranking time.",
         "python rank.py --candidates candidates.jsonl --out submission.csv"),
        ("Every score is inspectable.",
         "Interpretable sub-scores per candidate — easy to defend in the 'walk through your architecture' interview."),
        ("No LLM at inference → scales to a 200K pool in production.",
         "Exactly the latency/quality thinking the JD asks a Senior AI Engineer to demonstrate."),
        ("Honest, varied, fact-grounded reasoning passes manual review.",
         "Specific facts + JD connection + acknowledged concerns + rank-consistent tone."),
    ])
    d.save()
    print(f"[deck] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", default="outputs/Redrob_Approach_Deck.pdf")
    ap.add_argument("--team", default="Mann Sutaria")
    ap.add_argument("--date", default="June 2026")
    a = ap.parse_args()
    st = stats(a.candidates, a.submission)
    build(st, a.out, a.team, a.date)
