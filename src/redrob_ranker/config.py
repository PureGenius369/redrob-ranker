"""
config.py — The JD knowledge base.

This module encodes *what the "Senior AI Engineer — Founding Team" role actually
means*, translated from the natural-language job description into structured,
machine-usable taxonomies, evidence patterns, and scoring weights.

Design principle: the JD is deliberately written to defeat keyword matching. So
we don't match keywords — we encode the JD's *intent*:

  - what the role genuinely needs   (must-haves)
  - what counts as real evidence    (career-history phrases, not skill tags)
  - what the JD explicitly rejects   (disqualifiers / anti-patterns)
  - how to read between the lines    (product>services, applied>research,
                                      NLP/IR>CV/speech, available>dormant)

Everything here is interpretable on purpose: the same structures drive the
ranker, the per-candidate reasoning, the deck, and the "defend-your-work"
interview answers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. The JD, distilled into embedding "anchor" texts.
#    We embed candidate profiles against a POSITIVE anchor (the ideal hire) and
#    a NEGATIVE anchor (the explicitly-unwanted profile), then use the
#    difference (pos_sim - neg_sim) as the semantic signal. This contrastive
#    framing is what lets us surface "plain-language Tier-5s" who never say
#    "RAG" or "Pinecone" but describe building the systems.
# ---------------------------------------------------------------------------

JD_POSITIVE_ANCHOR = (
    "Senior AI engineer with six to eight years of experience, four to five of "
    "them in applied machine learning at product companies, not services or "
    "consulting. Has personally shipped an end-to-end search, ranking, "
    "retrieval, or recommendation system to real users at meaningful scale. "
    "Production experience with embeddings-based retrieval using "
    "sentence-transformers, BGE, E5, or OpenAI embeddings, and with vector "
    "databases or hybrid search such as FAISS, Pinecone, Weaviate, Qdrant, "
    "Milvus, OpenSearch, or Elasticsearch. Handles embedding drift, index "
    "refresh, and retrieval-quality regression in production. Strong Python and "
    "writes production code. Designs rigorous evaluation for ranking systems "
    "using NDCG, MRR, MAP, offline-to-online correlation, and A/B testing. "
    "Strong natural language processing and information retrieval background. "
    "Scrappy product-engineering attitude, ships fast, has strong defensible "
    "opinions about hybrid versus dense retrieval and when to fine-tune versus "
    "prompt. Understood retrieval and ranking before large language models "
    "became fashionable. Based in or willing to relocate to Pune or Noida, "
    "India, and actively available on the hiring platform."
)

JD_NEGATIVE_ANCHOR = (
    "A profile that lists many trendy AI keywords as skills but whose actual job "
    "title and career history are in an unrelated field such as marketing, "
    "human resources, sales, accounting, operations, content writing, graphic "
    "design, civil, or mechanical engineering. A pure academic or research-only "
    "background with no production deployment. Someone whose only AI experience "
    "is recent LangChain wrappers calling the OpenAI API with no earlier "
    "machine-learning production work. A title-chaser who switches companies "
    "every eighteen months for a bigger title. A career spent entirely at IT "
    "services and consulting firms. Primary expertise in computer vision, "
    "speech, audio, or robotics with no natural language or information "
    "retrieval experience. A senior person who stopped writing code years ago "
    "and only does architecture. A candidate who has been dormant for months "
    "and ignores recruiter messages."
)

# A short, human-readable role summary reused in reasoning and the deck.
ROLE_LABEL = "Senior AI Engineer (Founding Team) — Redrob AI"

# ---------------------------------------------------------------------------
# 2. Skill taxonomies.
#    Skills are a *claimed* field and are heavily stuffed in this dataset, so we
#    never trust a raw skill list. We bucket skills into meaning groups and only
#    ever use them after applying the trust weighting in structured.py
#    (endorsements x duration x proficiency x assessment score).
#    All matching is done case-insensitively on normalized skill names.
# ---------------------------------------------------------------------------

# The heart of the role: retrieval / ranking / search / recsys.
RETRIEVAL_RANKING_SKILLS = {
    "information retrieval", "semantic search", "vector search", "retrieval",
    "ranking", "learning to rank", "learning-to-rank", "ltr", "recommendation",
    "recommender systems", "recommendation systems", "recsys", "search",
    "search relevance", "elasticsearch", "opensearch", "solr", "lucene",
    "rag", "retrieval augmented generation", "retrieval-augmented generation",
    "hybrid search", "bm25", "okapi bm25", "reranking", "re-ranking",
}

VECTOR_DB_SKILLS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "vespa", "pgvector",
    "chroma", "chromadb", "annoy", "hnsw", "scann", "vald", "lancedb",
    "vector database", "vector db", "ann", "approximate nearest neighbor",
}

EMBEDDING_SKILLS = {
    "embeddings", "embedding", "sentence-transformers", "sentence transformers",
    "sbert", "bge", "e5", "gte", "bm25", "word2vec", "glove", "fasttext",
    "openai embeddings", "instructor", "nomic", "contrastive learning",
    "dense retrieval", "dpr", "colbert",
}

NLP_SKILLS = {
    "nlp", "natural language processing", "transformers", "bert", "roberta",
    "llm", "llms", "large language models", "language models", "tokenization",
    "text classification", "named entity recognition", "ner", "question answering",
    "summarization", "text mining", "spacy", "huggingface", "hugging face",
    "fine-tuning llms", "fine tuning", "fine-tuning", "lora", "qlora", "peft",
    "prompt engineering", "rlhf", "instruction tuning", "gpt", "t5", "llama",
}

LLM_FINETUNE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning", "fine tuning",
    "rlhf", "instruction tuning", "dpo", "sft", "adapter tuning",
}

MLOPS_SKILLS = {
    "mlflow", "weights & biases", "wandb", "weights and biases", "kubeflow",
    "airflow", "bentoml", "ray", "triton", "torchserve", "vertex ai",
    "sagemaker", "model serving", "feature store", "feast", "dvc", "onnx",
    "tensorrt", "model monitoring", "ml pipelines",
}

CORE_ML_SKILLS = {
    "machine learning", "deep learning", "pytorch", "tensorflow", "keras",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "feature engineering", "statistical modeling", "ml", "neural networks",
    "model training", "hyperparameter tuning", "gradient boosting",
}

# Down-weighted by the JD: "primary expertise in CV/speech/robotics without
# significant NLP/IR exposure". Presence is fine; *dominance without NLP* is the
# negative signal (handled in structured.py).
CV_SPEECH_ROBOTICS_SKILLS = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "opencv", "yolo", "gans", "gan", "stable diffusion",
    "image generation", "ocr", "speech recognition", "asr", "tts",
    "text to speech", "text-to-speech", "speech synthesis", "audio processing",
    "wav2vec", "robotics", "ros", "slam", "motion planning", "lidar",
    "pose estimation", "face recognition", "video analytics",
}

# "AI core skills" — the loose bucket a naive keyword matcher would count (this
# is roughly what the deliberately-bad sample_submission counted). We track it
# ONLY to detect stuffing (many of these as untrusted claims = red flag), never
# to reward.
AI_BUZZWORD_SKILLS = (
    RETRIEVAL_RANKING_SKILLS | VECTOR_DB_SKILLS | EMBEDDING_SKILLS
    | NLP_SKILLS | LLM_FINETUNE_SKILLS | CORE_ML_SKILLS | CV_SPEECH_ROBOTICS_SKILLS
)

# ---------------------------------------------------------------------------
# 3. Title taxonomy (current_title + career titles).
#    Title coherence is the single most decisive anti-keyword-stuffer signal:
#    the trap candidates have "Marketing Manager"/"HR Manager" titles with AI
#    skills stuffed in. We score title fit from these keyword buckets.
#    Matching: lowercase substring against the title string.
# ---------------------------------------------------------------------------

# Bullseye for THIS JD: retrieval / ranking / search / recsys / NLP / applied ML.
# These titles map directly onto "own the intelligence layer: ranking, retrieval,
# matching". (Exact strings observed in the data.)
TITLE_BULLSEYE = (
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied ml engineer", "applied scientist", "nlp engineer",
    "search engineer", "ranking engineer", "relevance engineer",
    "recommendation systems engineer", "recommendation engineer", "recsys",
    "(ml)",  # e.g. "Senior Software Engineer (ML)"
)

# Strong general AI/ML titles (not as precisely on-role as bullseye).
TITLE_CORE_AI = (
    "data scientist", "machine learning scientist", "ml scientist",
    "deep learning engineer", "ai specialist", "applied ai",
    "ml platform", "ml infra", "mlops engineer",
)

# AI-adjacent titles that carry a JD caveat (handled via domain/flags):
#   computer vision engineer  -> CV without NLP/IR is down-weighted
#   ai research engineer / research scientist -> research-only is a disqualifier
TITLE_CAUTION_AI = (
    "computer vision engineer", "ai research engineer", "research engineer",
    "research scientist", "ai researcher",
)

# Generic engineering/data titles. A *plain-language Tier-5* can live here:
# someone titled "Software Engineer" or "Data Engineer" whose career history
# shows they built search/ranking/recsys. career_evidence + semantics rescue them.
TITLE_ADJACENT = (
    "software engineer", "backend engineer", "data engineer", "sde",
    "full stack", "fullstack", "platform engineer", "analytics engineer",
    "software developer", "data analyst", "computer scientist",
    "staff engineer", "principal engineer", "senior engineer",
    "engineering manager", "tech lead", "solutions engineer",
    "cloud engineer", "devops engineer", "java developer", ".net developer",
    "mobile developer", "frontend engineer", "qa engineer",
)

# Titles that signal a non-AI professional. A profile whose *recent* titles are
# dominated by these, regardless of stuffed skills, is the keyword-stuffer trap.
TITLE_NON_TECH = (
    "marketing", "hr ", "human resources", "recruiter", "talent acquisition",
    "sales", "account executive", "accountant", "finance", "operations manager",
    "content writer", "copywriter", "editor", "graphic designer", "designer",
    "ux", "ui designer", "civil engineer", "mechanical engineer",
    "electrical engineer", "project manager", "product manager",
    "customer support", "customer success", "business analyst",
    "business development", "consultant", "teacher", "professor", "lecturer",
    "administrator", "coordinator", "executive assistant", "supply chain",
    "logistics", "procurement", "legal", "paralegal", "nurse", "doctor",
)

# ---------------------------------------------------------------------------
# 4. Career-evidence phrase patterns.
#    These are scanned over career-history DESCRIPTIONS + summary (the fields
#    where real work shows up). Evidence in a description is worth far more than
#    a matching skill tag. Patterns are plain lowercase substrings; structured.py
#    counts how many *distinct evidence categories* a candidate demonstrates.
# ---------------------------------------------------------------------------

EVIDENCE_RETRIEVAL = (
    "semantic search", "vector search", "embedding", "embeddings", "retrieval",
    "rag", "retrieval-augmented", "retrieval augmented", "nearest neighbor",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "dense retrieval", "hybrid search", "bm25", "search index",
    "vector database", "knn", "ann index",
)

EVIDENCE_RANKING_RECSYS = (
    "ranking", "rank", "learning to rank", "learning-to-rank", "recommendation",
    "recommender", "recsys", "relevance", "search relevance", "re-rank",
    "reranking", "personalization", "matching system", "candidate matching",
    "feed ranking", "ctr prediction",
)

EVIDENCE_PRODUCTION = (
    "production", "deployed", "deployment", "real users", "in production",
    "at scale", "shipped", "launched", "served", "serving", "latency",
    "throughput", "millions of", "live traffic", "online system",
    "low-latency", "real-time", "real time", "end-to-end", "end to end",
)

EVIDENCE_EVAL = (
    "ndcg", "mrr", "map@", "mean average precision", "precision@",
    "recall@", "a/b test", "a/b testing", "ab test", "offline evaluation",
    "online evaluation", "offline-to-online", "offline to online", "auc",
    "evaluation framework", "eval harness", "benchmark", "metrics", "ndcg@",
)

EVIDENCE_NLP_IR = (
    "nlp", "natural language", "language model", "llm", "transformer", "bert",
    "text classification", "named entity", "question answering", "summariz",
    "information retrieval", "tokeniz", "fine-tun", "fine tun", "embeddings",
)

# Negative-evidence patterns (disqualifiers from the JD).
EVIDENCE_RESEARCH_ONLY = (
    "research lab", "academic research", "phd research", "postdoc",
    "research assistant", "published papers", "publications", "research-only",
    "purely research", "university research", "research intern", "thesis",
)

EVIDENCE_LANGCHAIN_WRAPPER = (
    "langchain", "llamaindex", "llama index", "auto-gpt", "autogpt",
    "prompt chaining", "openai api", "gpt wrapper", "chatgpt plugin",
)

EVIDENCE_NO_CODE_LEADERSHIP = (
    "no longer code", "stepped away from coding", "purely architectural",
    "moved into architecture", "tech lead role", "people management",
    "managed a team", "led a team of",
)

# ---------------------------------------------------------------------------
# 5. Company taxonomy (product vs services).
#    JD disqualifier: "only worked at consulting firms (TCS, Infosys, Wipro,
#    Accenture, Cognizant, Capgemini, etc.) in their entire career." Note: this
#    is only a disqualifier if it's the *entire* career. Any product-company
#    stint mitigates it.
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "hcl technologies", "ltimindtree",
    "mindtree", "lti", "larsen & toubro infotech", "mphasis", "dxc",
    "ibm services", "deloitte", "pwc", "kpmg", "ernst & young", "ey",
    "genpact", "wns", "hexaware", "birlasoft", "coforge", "persistent systems",
    "cybage", "zensar", "virtusa", "ust global", "nttdata", "ntt data",
}

# Well-known product / tech companies (small curated set; used as a positive
# nudge, not a gate). Detection is generous because the dataset uses both real
# and fictional company names.
PRODUCT_COMPANY_HINTS = {
    "google", "meta", "facebook", "amazon", "microsoft", "apple", "netflix",
    "uber", "airbnb", "linkedin", "twitter", "stripe", "flipkart", "swiggy",
    "zomato", "ola", "paytm", "phonepe", "razorpay", "cred", "meesho",
    "myntra", "nvidia", "openai", "anthropic", "cohere", "databricks",
    "snowflake", "spotify", "pinterest", "dropbox", "atlassian", "adobe",
    "salesforce", "nutanix", "freshworks", "zoho", "postman", "browserstack",
    "sprinklr", "rubrik", "intuit", "walmart", "target", "shopify",
}

# ---------------------------------------------------------------------------
# 6. Location taxonomy.
#    JD: Pune/Noida-preferred; Hyderabad, Mumbai, Delhi NCR, Bangalore welcome;
#    open to relocation candidates from Tier-1 Indian cities; outside India
#    case-by-case with no visa sponsorship.
# ---------------------------------------------------------------------------

PREFERRED_CITIES = {"pune", "noida"}

INDIA_TIER1_CITIES = {
    "pune", "noida", "bangalore", "bengaluru", "hyderabad", "mumbai",
    "delhi", "new delhi", "gurgaon", "gurugram", "ncr", "chennai",
    "kolkata", "ahmedabad",
}

INDIA_MARKERS = {"india", "bharat"}

# ---------------------------------------------------------------------------
# 7. Scoring weights.
#    Tunable; defaults chosen so that *title/career coherence + real evidence*
#    dominate, semantic similarity supports (esp. for plain-language Tier-5s),
#    and behavioral signals modulate availability. Calibrated against EDA.
# ---------------------------------------------------------------------------

# Structured-fit component weights (sum is normalized internally).
STRUCT_WEIGHTS = {
    "role_title": 0.26,      # current/recent title coherence with an AI role
    "career_evidence": 0.30, # demonstrated retrieval/ranking/production/eval work
    "experience": 0.10,      # 5-9 yrs band (ideal 6-8)
    "skill_trust": 0.12,     # trust-weighted AI-skill mass (anti-stuffing)
    "domain_nlp_ir": 0.10,   # NLP/IR positive; CV/speech-only negative
    "product_vs_services": 0.06,
    "location": 0.06,
}

# How the final score blends structured fit with the contrastive semantic signal.
BLEND_SEMANTIC = 0.30
BLEND_STRUCTURED = 0.70

# Behavioral modifier clamps (multiplier applied to the blended fit score).
BEHAVIOR_MIN_MULT = 0.55
BEHAVIOR_MAX_MULT = 1.12

# Honeypot / hard-disqualifier gate: profiles flagged impossible get multiplied
# by this (effectively removed from the top-100 region).
HONEYPOT_GATE_MULT = 0.02

# Reference "today" for recency/date math. Overridable from the CLI so the
# ranking is reproducible regardless of wall-clock at run time.
REFERENCE_DATE = "2026-06-21"

# Experience band.
EXP_IDEAL_LOW, EXP_IDEAL_HIGH = 6.0, 8.0
EXP_SOFT_LOW, EXP_SOFT_HIGH = 5.0, 9.0
