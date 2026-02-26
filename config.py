"""
config.py — ResumeRank AI v2
Centralized configuration: weight profiles, skill taxonomy, thresholds
"""

# ── Weight Profiles (recruiter selects via UI) ──────────────────────────────
WEIGHT_PROFILES = {
    "balanced": {
        "label": "Balanced",
        "description": "General purpose — equal emphasis on skills, similarity, experience",
        "weights": {
            "tfidf_similarity": 0.30,
            "keyword_match":    0.25,
            "skill_overlap":    0.25,
            "experience":       0.12,
            "education":        0.08,
        }
    },
    "senior": {
        "label": "Senior / Lead",
        "description": "Prioritizes experience and leadership indicators",
        "weights": {
            "tfidf_similarity": 0.25,
            "keyword_match":    0.20,
            "skill_overlap":    0.20,
            "experience":       0.25,
            "education":        0.10,
        }
    },
    "fresher": {
        "label": "Fresh Graduate",
        "description": "Prioritizes education, projects and foundational skills",
        "weights": {
            "tfidf_similarity": 0.30,
            "keyword_match":    0.25,
            "skill_overlap":    0.25,
            "experience":       0.05,
            "education":        0.15,
        }
    },
    "technical": {
        "label": "Technical / Engineering",
        "description": "Maximum weight on technical skills and stack overlap",
        "weights": {
            "tfidf_similarity": 0.20,
            "keyword_match":    0.20,
            "skill_overlap":    0.40,
            "experience":       0.12,
            "education":        0.08,
        }
    },
    "management": {
        "label": "Management / MBA",
        "description": "Emphasizes education, domain keywords and soft skills",
        "weights": {
            "tfidf_similarity": 0.30,
            "keyword_match":    0.30,
            "skill_overlap":    0.15,
            "experience":       0.15,
            "education":        0.10,
        }
    }
}

DEFAULT_PROFILE = "balanced"

# ── Grade Thresholds ─────────────────────────────────────────────────────────
GRADE_SCALE = [
    (85, "A+", "Outstanding Match"),
    (75, "A",  "Excellent Match"),
    (65, "B",  "Good Match"),
    (50, "C",  "Moderate Match"),
    (35, "D",  "Weak Match"),
    (0,  "F",  "Poor Match"),
]

# ── Skill Taxonomy (tiered by importance) ───────────────────────────────────
# Tier 1: Core/hard skills — highest weight in skill scoring
TIER1_SKILLS = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "swift", "kotlin", "scala", "r", "matlab", "julia",
    # ML/AI
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    "hugging face", "transformers", "bert", "gpt", "llm", "langchain",
    "computer vision", "nlp", "deep learning", "machine learning", "reinforcement learning",
    # Data
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery",
    # Cloud/DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    # Web
    "react", "node", "django", "flask", "fastapi", "spring", "graphql",
}

# Tier 2: Tools and platforms — medium weight
TIER2_SKILLS = {
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau",
    "power bi", "excel", "git", "github", "jira", "confluence",
    "mlflow", "wandb", "dvc", "opencv", "nltk", "spacy",
    "html", "css", "sass", "webpack", "linux", "bash",
    "agile", "scrum", "kanban", "microservices", "rest api", "graphql",
    "jupyter", "colab", "databricks", "sagemaker",
}

# Tier 3: Soft/general skills — lower weight
TIER3_SKILLS = {
    "leadership", "communication", "teamwork", "problem solving", "analytical",
    "critical thinking", "project management", "stakeholder management",
    "data analysis", "statistics", "mathematics", "research",
}

ALL_SKILLS = TIER1_SKILLS | TIER2_SKILLS | TIER3_SKILLS

SKILL_WEIGHTS = {s: 3.0 for s in TIER1_SKILLS}
SKILL_WEIGHTS.update({s: 2.0 for s in TIER2_SKILLS})
SKILL_WEIGHTS.update({s: 1.0 for s in TIER3_SKILLS})

# ── Education Level Map ──────────────────────────────────────────────────────
EDUCATION_LEVELS = {
    # Doctorate
    r'\bph\.?d\b': 1.00, r'\bdoctorate\b': 1.00, r'\bd\.sc\b': 0.98,
    # Masters
    r'\bm\.?tech\b': 0.88, r'\bm\.?e\b': 0.87, r'\bmtech\b': 0.88,
    r'\bmaster\b': 0.87, r'\bm\.?s\b': 0.86, r'\bm\.?sc\b': 0.85,
    r'\bmba\b': 0.84, r'\bm\.?eng\b': 0.86,
    # Bachelors
    r'\bb\.?tech\b': 0.72, r'\bb\.?e\b': 0.71, r'\bbtech\b': 0.72,
    r'\bbachelor\b': 0.70, r'\bb\.?s\b': 0.68, r'\bb\.?sc\b': 0.67,
    r'\bb\.?eng\b': 0.70, r'\bb\.?a\b': 0.65,
    # Others
    r'\bassociate\b': 0.50, r'\bdiploma\b': 0.42, r'\bcertif\b': 0.35,
}

# ── Experience: keyword weights for date-range detection ────────────────────
EXPERIENCE_SECTION_HEADERS = [
    "experience", "employment", "work history", "professional background",
    "career history", "positions held"
]

# ── Keyword Stuffing: max useful frequency per token ────────────────────────
MAX_TOKEN_FREQUENCY = 5   # occurrences beyond this are capped

# ── File handling ────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx'}
MAX_FILE_SIZE_MB = 10
AUTO_DELETE_UPLOADS = True   # delete file after scoring

# ── Section headers for section-aware scoring ───────────────────────────────
SKILL_SECTION_HEADERS = [
    "skills", "technical skills", "core competencies", "technologies",
    "tools", "languages", "frameworks", "expertise", "proficiencies",
    "tech stack", "technical expertise", "key skills", "competencies"
]
