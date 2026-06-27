"""
Build the approach deck as a PDF (landscape slides) with reportlab — no external
converter needed. Reads the final submission CSV + candidate pool to embed real
top-100 statistics.

    python scripts/build_deck.py --candidates ./candidates.jsonl \
        --submission ./submission.csv --out ./deck/Redrob_Approach_Deck.pdf
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

# --- Warm earth palette (parchment paper, espresso ink, terracotta/ochre/olive) ---
PAPER = HexColor("#F3E7D1")   # warm parchment / sand background
CARD = HexColor("#EAD9BB")    # warm tan card fill
LINE = HexColor("#DBC8A4")    # warm hairline
INK = HexColor("#352A1B")     # warm espresso — titles, dark cover
BODY = HexColor("#4E4230")    # warm brown body text
MUTED = HexColor("#937F5E")   # warm taupe captions
CREAM = HexColor("#F6EDD8")   # text on dark cover
WHITE = HexColor("#FFFFFF")

AMBER = HexColor("#C45B26")   # terracotta — primary accent
TEAL = HexColor("#6E7A3C")    # olive — secondary accent
GOLD = HexColor("#BE8420")    # ochre
BRICK = HexColor("#A8402A")   # rust
STEEL = HexColor("#9A6132")   # warm clay/brown (structured box)

W, H = landscape(A4)  # 842 x 595


def stats(candidates_path, submission_path):
    rows = []
    with open(submission_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
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
        "n": n, "titles": titles.most_common(8),
        "india_pct": round(100 * india / n), "otw_pct": round(100 * otw / n),
        "mean_yoe": round(yoe_sum / n, 1), "honeypots": hp,
        "top5": rows[:5], "score_hi": rows[0]["score"], "score_lo": rows[-1]["score"],
    }


class Deck:
    def __init__(self, path):
        self.c = canvas.Canvas(path, pagesize=landscape(A4))
        self.page = 0

    def _footer(self):
        c = self.c
        c.setStrokeColor(LINE)
        c.setLineWidth(0.8)
        c.line(18 * mm, 12 * mm, W - 18 * mm, 12 * mm)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(MUTED)
        c.drawString(18 * mm, 8 * mm, "Redrob · Intelligent Candidate Discovery & Ranking")
        c.setFillColor(AMBER)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawRightString(W - 18 * mm, 8 * mm, f"{self.page:02d}")

    def header(self, kicker, title):
        """Light editorial header: amber kicker, slate title, short accent rule."""
        self.page += 1
        c = self.c
        c.setFillColor(PAPER)
        c.rect(0, 0, W, H, fill=1, stroke=0)
        # three small accent ticks (brand mark)
        for i, col in enumerate((AMBER, TEAL, GOLD)):
            c.setFillColor(col)
            c.rect(18 * mm + i * 4.6 * mm, H - 19 * mm, 3.4 * mm, 3.4 * mm, fill=1, stroke=0)
        c.setFillColor(AMBER)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(18 * mm, H - 26 * mm, kicker.upper())
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(18 * mm, H - 37 * mm, title)
        c.setStrokeColor(AMBER)
        c.setLineWidth(2.4)
        c.line(18 * mm, H - 41 * mm, 42 * mm, H - 41 * mm)
        self._footer()

    def bullets(self, items, x=20 * mm, y=H - 54 * mm, dy=13 * mm, size=12.5):
        c = self.c
        for it in items:
            head, sub = it if isinstance(it, tuple) else (it, None)
            c.setFillColor(AMBER)
            c.rect(x, y + 0.4 * mm, 2.4 * mm, 2.4 * mm, fill=1, stroke=0)  # square marker
            c.setFont("Helvetica-Bold", size)
            c.setFillColor(INK)
            c.drawString(x + 6 * mm, y, head)
            if sub:
                c.setFont("Helvetica", size - 2.5)
                c.setFillColor(BODY)
                c.drawString(x + 6 * mm, y - 5.6 * mm, sub)
                y -= dy + 4 * mm
            else:
                y -= dy
        return y

    def _wrap(self, text, font, size, max_w):
        """Greedy word-wrap to a max pixel width (prevents column overlap)."""
        c = self.c
        line, out = "", []
        for w_ in text.split():
            trial = (line + " " + w_).strip()
            if c.stringWidth(trial, font, size) <= max_w:
                line = trial
            else:
                if line:
                    out.append(line)
                line = w_
        if line:
            out.append(line)
        return out

    def col_bullets(self, items, x, y, max_w, size=11.5,
                    line_gap=5.4 * mm, item_gap=4.8 * mm):
        """Bulleted list confined to a column of width max_w; wraps long lines
        so a wide left column never collides with the right column."""
        c = self.c
        font = "Helvetica-Bold"
        for it in items:
            c.setFillColor(AMBER)
            c.rect(x, y + 0.4 * mm, 2.4 * mm, 2.4 * mm, fill=1, stroke=0)
            c.setFont(font, size)
            c.setFillColor(INK)
            for ln in self._wrap(it, font, size, max_w):
                c.drawString(x + 6 * mm, y, ln)
                y -= line_gap
            y -= item_gap
        return y

    def box(self, x, y, w, h, fill, text, sub=None):
        c = self.c
        c.setFillColor(fill)
        c.roundRect(x, y, w, h, 3 * mm, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 11.5)
        c.drawCentredString(x + w / 2, y + h - 9 * mm, text)
        c.setStrokeColor(HexColor("#FFFFFF"))
        c.setLineWidth(0.5)
        c.setFillColor(CREAM)
        c.setFont("Helvetica", 8)
        ty = y + h - 15 * mm
        for line in (sub or []):
            c.drawCentredString(x + w / 2, ty, line)
            ty -= 4.4 * mm

    def save(self):
        self.c.showPage()
        self.c.save()


def build(st, out, team, date):
    d = Deck(out)
    c = d.c

    # ---- Slide 1: cover (dark) ----
    d.page += 1
    c.setFillColor(INK)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    for i, col in enumerate((AMBER, TEAL, GOLD, BRICK)):
        c.setFillColor(col)
        c.rect(20 * mm + i * 7 * mm, H - 34 * mm, 5 * mm, 5 * mm, fill=1, stroke=0)
    c.setFillColor(AMBER)
    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(20 * mm, H - 46 * mm, "DATA & AI CHALLENGE  ·  INTELLIGENT CANDIDATE DISCOVERY & RANKING")
    c.setFillColor(CREAM)
    c.setFont("Helvetica-Bold", 40)
    c.drawString(20 * mm, H - 78 * mm, "Reading the profile,")
    c.setFillColor(AMBER)
    c.drawString(20 * mm, H - 95 * mm, "not the keywords.")
    c.setFillColor(HexColor("#AEB9C2"))
    c.setFont("Helvetica", 14)
    c.drawString(20 * mm, H - 112 * mm, "A hybrid, CPU-only ranker that beats keyword-stuffers,")
    c.drawString(20 * mm, H - 120 * mm, "surfaces plain-language talent, and avoids the honeypots.")
    c.setFillColor(CREAM)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 26 * mm, team)
    c.showPage()

    # ---- Slide 2: problem ----
    d.header("The problem", "Keyword filters can't see what matters")
    d.bullets([
        ("Recruiters miss the right person — keyword filters are blind to real fit.",
         "The JD is written to defeat keyword matching; the right answer is reasoning about what it means."),
        ("The dataset is adversarial by design.",
         "Keyword stuffers, plain-language strong candidates, behavioral twins, and ~80 'impossible' honeypots."),
        ("The shipped sample submission deliberately falls for it.",
         "It ranks an HR Manager #1 and real ML Engineers at #27 / #48 / #99 — purely on AI-skill counts."),
        ("Goal: a shortlist a recruiter can trust — that survives reproduction + interview.",
         "NDCG@10 dominates scoring (0.50), so the top-10 must be genuinely excellent."),
    ])
    c.showPage()

    # ---- Slide 3: what the JD means ----
    d.header("Reading between the lines", "What the role actually needs")
    LX, RX, LW, RW = 20 * mm, 120 * mm, 86 * mm, 150 * mm
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 12.5)
    c.drawString(LX, H - 52 * mm, "FIT — the ideal hire")
    d.col_bullets([
        "6–8 yrs, applied ML at product companies (not services)",
        "Shipped search / ranking / retrieval / recsys to real users",
        "Embeddings + vector/hybrid search + rigorous ranking eval",
        "Strong NLP/IR; in/near Pune–Noida; actually available",
    ], x=LX, y=H - 62 * mm, max_w=LW)
    c.setFillColor(BRICK)
    c.setFont("Helvetica-Bold", 12.5)
    c.drawString(RX, H - 52 * mm, "ANTI-FIT — explicit disqualifiers")
    d.col_bullets([
        "Stuffed AI skills but a non-technical real title",
        "Pure research / academia with no production",
        "Only recent LangChain-on-OpenAI wrappers",
        "CV/speech-only · services-only · title-chasers · dormant",
    ], x=RX, y=H - 62 * mm, max_w=RW)
    c.showPage()

    # ---- Slide 4: architecture ----
    d.header("Architecture", "Four interpretable layers, no LLM at inference")
    c.setFillColor(CARD)
    c.roundRect(20 * mm, H - 72 * mm, W - 40 * mm, 16 * mm, 3 * mm, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("Courier-Bold", 11.5)
    c.drawCentredString(W / 2, H - 63.5 * mm,
                        "score = (0.70 structured_fit + 0.30 semantic_fit) x behavioral_modifier x honeypot_gate")
    bw, bh, by = 38 * mm, 33 * mm, H - 118 * mm
    xs = [20 * mm, 70 * mm, 120 * mm, 170 * mm]
    d.box(xs[0], by, bw, bh, STEEL, "Structured fit", ["title coherence", "career evidence", "skill trust", "domain · location"])
    d.box(xs[1], by, bw, bh, TEAL, "Semantic fit", ["contrastive:", "JD-ideal minus", "JD-anti-pattern", "static embeds"])
    d.box(xs[2], by, bw, bh, GOLD, "Behavioral", ["availability", "recency · response", "open-to-work", "notice period"])
    d.box(xs[3], by, bw, bh, BRICK, "Honeypot gate", ["impossible", "tenure / skills", "removed from", "top-100"])
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(20 * mm, by - 9 * mm,
                 "Two-pass: scalar scoring over 100K, then full facts + grounded reasoning for the top 100. CPU-only, streams the pool.")
    c.showPage()

    # ---- Slide 5: structured ----
    d.header("Layer 1 — Structured fit", "The JD's logic, made machine-usable (70% weight)")
    d.bullets([
        ("Title coherence — the decisive anti-stuffer signal.",
         "A 'Marketing Manager' with 9 AI skills scores ~0.06; bullseye AI/search/recsys titles score 1.0."),
        ("Career evidence — read from descriptions, not skill tags.",
         "Credits demonstrated retrieval / ranking / production / NDCG-MAP evaluation / NLP-IR work."),
        ("Skill trust — defeats stuffing.",
         "Skills reweighted by endorsements x duration x proficiency x platform assessment (raw lists ignored)."),
        ("Plus experience band, product-vs-services, NLP/IR-vs-CV/speech, location, disqualifier penalties.",
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
         "No transformer forward pass, no torch, no GPU: the latency/quality tradeoff the JD asks about."),
        ("Pure scikit-learn LSA fallback keeps it reproducible with zero downloads.", None),
    ])
    c.showPage()

    # ---- Slide 7: behavioral + honeypot ----
    d.header("Layers 3 & 4 — Available & possible", "Down-weight the unavailable; remove the impossible")
    LX, RX, LW, RW = 20 * mm, 120 * mm, 86 * mm, 150 * mm
    c.setFillColor(GOLD)
    c.setFont("Helvetica-Bold", 12.5)
    c.drawString(LX, H - 52 * mm, "Behavioral modifier")
    d.col_bullets([
        "'Dormant 6 months + 5% response = not available.'",
        "Bounded multiplier from 23 signals, calibrated to the pool.",
        "Recency, response rate, open-to-work, notice, demand.",
    ], x=LX, y=H - 62 * mm, max_w=LW)
    c.setFillColor(BRICK)
    c.setFont("Helvetica-Bold", 12.5)
    c.drawString(RX, H - 52 * mm, "Honeypot gate")
    d.col_bullets([
        "Detects internal impossibilities, not specific IDs.",
        "Tenure > time elapsed; ≥3 expert skills at 0 months.",
        f"{st['honeypots']} honeypots in our top-100 (cap is 10%).",
    ], x=RX, y=H - 62 * mm, max_w=RW)
    c.showPage()

    # ---- Slide 8: EDA ----
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
    cards = [(f"{st['india_pct']}%", "based in India"), (f"{st['otw_pct']}%", "open to work"),
             (f"{st['mean_yoe']}", "mean years exp."), (f"{st['honeypots']}", "honeypots in top-100")]
    accents = [TEAL, GOLD, STEEL, BRICK]
    cx = 20 * mm
    for (big, lab), col in zip(cards, accents):
        c.setFillColor(CARD)
        c.roundRect(cx, H - 74 * mm, 44 * mm, 23 * mm, 3 * mm, fill=1, stroke=0)
        c.setFillColor(col)
        c.rect(cx, H - 74 * mm, 2.6 * mm, 23 * mm, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 23)
        c.drawString(cx + 7 * mm, H - 63 * mm, big)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(cx + 7 * mm, H - 69.5 * mm, lab)
        cx += 49 * mm
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, H - 84 * mm, "Sample top-ranked reasoning (from the candidate's own facts — no hallucination):")
    y = H - 92 * mm
    for row in st["top5"][:4]:
        c.setFillColor(AMBER)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(20 * mm, y, f"#{row['rank']}")
        c.setFillColor(BODY)
        c.setFont("Helvetica", 8.4)
        words, line, lines = row["reasoning"].split(), "", []
        for w_ in words:
            if c.stringWidth(line + " " + w_, "Helvetica", 8.4) < (W - 52 * mm):
                line = (line + " " + w_).strip()
            else:
                lines.append(line); line = w_
        lines.append(line)
        for ln in lines[:2]:
            c.drawString(29 * mm, y, ln); y -= 4.6 * mm
        y -= 2.8 * mm
    c.showPage()

    # ---- Slide 10: reproducibility ----
    d.header("Reproducible & defensible", "Built for Stage-3 reproduction and the interview")
    d.bullets([
        ("One command, CPU-only, no network at ranking time.",
         "python rank.py --candidates candidates.jsonl --out submission.csv  (~3 min, < 5-min budget)"),
        ("Every score is inspectable.",
         "Interpretable sub-scores per candidate — easy to defend in the architecture interview."),
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
    ap.add_argument("--out", default="deck/Redrob_Approach_Deck.pdf")
    ap.add_argument("--team", default="Mann Sutariya")
    ap.add_argument("--date", default="June 2026")
    a = ap.parse_args()
    st = stats(a.candidates, a.submission)
    build(st, a.out, a.team, a.date)
