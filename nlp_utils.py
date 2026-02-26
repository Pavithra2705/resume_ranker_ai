"""
nlp_utils.py — ResumeRank AI v2
NLP utilities: preprocessing, entity extraction, experience/education detection
"""

import re
import math
from config import (
    ALL_SKILLS, SKILL_WEIGHTS, TIER1_SKILLS, EDUCATION_LEVELS,
    SKILL_SECTION_HEADERS, EXPERIENCE_SECTION_HEADERS, MAX_TOKEN_FREQUENCY
)

# ── SpaCy (optional) ─────────────────────────────────────────────────────────
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    _nlp = None
    SPACY_OK = False

# ── Built-in stopword set ────────────────────────────────────────────────────
STOP_WORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her','it',
    'its','they','them','their','what','which','who','this','that','these',
    'those','am','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','a','an','the','and','but','if','or','as','of',
    'at','by','for','with','into','through','during','before','after','to',
    'from','up','in','out','on','over','then','here','there','when','where',
    'all','both','each','more','most','other','some','no','not','only','own',
    'same','so','than','too','very','can','will','just','should','now','also',
    'use','used','using','work','working','experience','years','year',
    'strong','good','excellent','ability','skills','skill','knowledge',
    'team','position','role','job','company','looking','seeking','candidate',
    'required','requirement','preferred','including','responsible','duties',
    'able','well','new','high','large','based','build','built','provide',
    'develop','developed','ensure','help','support','including','etc',
    'across','within','while','following','per','key','various','multiple',
    'cross','end','full','given','like','need','needs','overall','related',
    'specific','used','using','via','way','whether','without','would'
}

# ── Text normalization ────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Clean raw text: fix encoding artifacts, collapse whitespace."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)       # remove non-ASCII
    text = re.sub(r'\s+', ' ', text)                   # collapse whitespace
    text = re.sub(r'[•·▪▸►‣⁃◦‐–—]+', ' ', text)      # bullet chars → space
    return text.strip()


def simple_lemmatize(word: str) -> str:
    """Rule-based suffix stripping fallback lemmatizer."""
    rules = [
        (r'ing$', ''), (r'tions?$', ''), (r'ments?$', ''), (r'nesses?$', ''),
        (r'ically$', 'ic'), (r'ally$', 'al'), (r'edly$', ''), (r'ized?$', 'ize'),
        (r'ises?$', 'ise'), (r'ers?$', ''), (r'es$', ''), (r's$', ''),
    ]
    for pattern, repl in rules:
        result = re.sub(pattern, repl, word)
        if result != word and len(result) > 3:
            return result
    return word


def preprocess_text(text: str, anti_stuffing: bool = True) -> str:
    """
    Tokenize, lemmatize, remove stopwords.
    anti_stuffing: cap each token at MAX_TOKEN_FREQUENCY to prevent TF-IDF inflation.
    """
    text = normalize_text(text).lower()

    if SPACY_OK:
        doc = _nlp(text[:800000])
        tokens = [
            t.lemma_ for t in doc
            if not t.is_stop and not t.is_punct and t.is_alpha and len(t.text) > 2
        ]
    else:
        raw = re.findall(r'[a-zA-Z]+', text)
        tokens = [
            simple_lemmatize(w) for w in raw
            if w not in STOP_WORDS and len(w) > 2
        ]

    if anti_stuffing:
        from collections import Counter
        counts = Counter(tokens)
        tokens = []
        for tok, cnt in counts.items():
            tokens.extend([tok] * min(cnt, MAX_TOKEN_FREQUENCY))

    return " ".join(tokens)


# ── Skill extraction ──────────────────────────────────────────────────────────
def extract_skills(text: str) -> dict:
    """
    Returns:
        {
          "all": [...],          # all skills found
          "section_skills": [...], # skills found inside a skills section (bonus)
          "tier1": [...],        # high-value skills
        }
    """
    text_lower = text.lower()
    found_all = []
    found_tier1 = []

    for skill in ALL_SKILLS:
        # word-boundary safe pattern
        pattern = r'(?<![a-zA-Z])' + re.escape(skill) + r'(?![a-zA-Z])'
        if re.search(pattern, text_lower):
            found_all.append(skill)
            if skill in TIER1_SKILLS:
                found_tier1.append(skill)

    # Section-aware: extract skills within skill sections
    section_skills = _extract_from_section(text_lower, SKILL_SECTION_HEADERS)

    return {
        "all": found_all,
        "section_skills": section_skills,
        "tier1": found_tier1
    }


def _extract_from_section(text_lower: str, section_headers: list) -> list:
    """Extract skills found within a named section block."""
    found = []
    for header in section_headers:
        # Find the section
        pattern = rf'(?:^|\n)\s*{re.escape(header)}\s*[:\-\n]'
        match = re.search(pattern, text_lower)
        if match:
            # Grab up to 600 chars after the section header
            segment = text_lower[match.end(): match.end() + 600]
            # Stop at next section header (capitalized word followed by newline)
            next_sec = re.search(r'\n[A-Z][A-Z\s]{2,}\n', segment)
            if next_sec:
                segment = segment[:next_sec.start()]
            for skill in ALL_SKILLS:
                pat = r'(?<![a-zA-Z])' + re.escape(skill) + r'(?![a-zA-Z])'
                if re.search(pat, segment):
                    found.append(skill)
    return list(set(found))


# ── Experience extraction ─────────────────────────────────────────────────────
_WORD_NUMS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20,
}


def extract_years_of_experience(text: str) -> float:
    """
    Multi-strategy experience extractor:
      1. Explicit "X years of experience" phrases (incl. decimals + word numbers)
      2. Date range calculation: "2019 – 2024", "Jan 2020 – Present"
    Returns the best (max) estimate.
    """
    text_lower = text.lower()
    candidates = []

    # ── Strategy 1: explicit mentions ────────────────────────────────────────
    # Numeric: "5 years", "3.5 years", "5+ years"
    pat1 = r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp|work)'
    for m in re.finditer(pat1, text_lower):
        val = float(m.group(1))
        if 0 < val < 50:
            candidates.append(val)

    # experience of X years
    pat2 = r'(?:experience|exp)\s+(?:of\s+)?(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)'
    for m in re.finditer(pat2, text_lower):
        val = float(m.group(1))
        if 0 < val < 50:
            candidates.append(val)

    # "over five years", "more than 3 years"
    pat3 = r'(?:over|more\s+than|approximately|about|nearly|almost|around)\s+(\w+)\s+(?:years?|yrs?)'
    for m in re.finditer(pat3, text_lower):
        word = m.group(1)
        # numeric
        try:
            val = float(word)
            if 0 < val < 50:
                candidates.append(val)
        except ValueError:
            if word in _WORD_NUMS:
                candidates.append(float(_WORD_NUMS[word]))

    # Word number: "five years of experience"
    pat4 = r'\b(' + '|'.join(_WORD_NUMS.keys()) + r')\+?\s+(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)'
    for m in re.finditer(pat4, text_lower):
        candidates.append(float(_WORD_NUMS[m.group(1)]))

    # ── Strategy 2: date-range calculation ───────────────────────────────────
    year_now = 2025
    # Find patterns like "2019 – 2024", "2020 - present", "Jan 2018 – Dec 2022"
    date_range_pat = re.compile(
        r'(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?'
        r'((?:19|20)\d{2})'
        r'\s*[-–—to]+\s*'
        r'((?:19|20)\d{2}|present|current|now|till\s+date|ongoing)',
        re.IGNORECASE
    )
    for m in date_range_pat.finditer(text_lower):
        start = int(m.group(1))
        end_raw = m.group(2).strip()
        if re.match(r'\d{4}', end_raw):
            end = int(end_raw)
        else:
            end = year_now
        duration = end - start
        if 0 < duration < 50:
            candidates.append(float(duration))

    if not candidates:
        return 0.0

    # Use max (most experienced claim), capped reasonably
    return min(max(candidates), 40.0)


# ── Education extraction ──────────────────────────────────────────────────────
def extract_education_score(text: str) -> tuple:
    """
    Returns (score: float, best_label: str)
    Uses word-boundary regex to avoid false positives.
    """
    text_lower = text.lower()
    best_score = 0.0
    best_label = "Not specified"

    for pattern, score in EDUCATION_LEVELS.items():
        if re.search(pattern, text_lower):
            if score > best_score:
                best_score = score
                best_label = pattern.replace(r'\b', '').replace('\\', '').replace('?', '').replace('.', '').upper()

    return best_score, best_label


# ── Section parser ────────────────────────────────────────────────────────────
def extract_sections(text: str) -> dict:
    """
    Split resume into named sections.
    Returns dict of section_name → section_text.
    """
    sections = {}
    lines = text.split('\n')
    current_section = "header"
    buffer = []

    KNOWN_HEADERS = {
        'summary', 'objective', 'profile', 'about',
        'experience', 'work experience', 'employment', 'career',
        'education', 'academic', 'qualification',
        'skills', 'technical skills', 'competencies', 'expertise',
        'projects', 'achievements', 'accomplishments',
        'certifications', 'certificates', 'awards',
        'publications', 'research', 'languages', 'interests'
    }

    for line in lines:
        stripped = line.strip().lower()
        # Check if this line looks like a section header
        is_header = (
            stripped in KNOWN_HEADERS or
            any(stripped.startswith(h) for h in KNOWN_HEADERS)
        ) and len(stripped) < 60 and stripped != ''

        if is_header:
            if buffer:
                sections[current_section] = '\n'.join(buffer)
            current_section = stripped
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections[current_section] = '\n'.join(buffer)

    return sections


# ── Required skill extraction from JD ───────────────────────────────────────
def extract_required_skills(jd_text: str) -> dict:
    """
    Parses the JD to split skills into:
      - required: explicitly flagged as must-have
      - preferred: nice-to-have / bonus
      - all: everything detected

    Detection strategy:
      1. Find "required skills" / "must have" sections → all skills inside = required
      2. Find "preferred" / "nice to have" sections → skills inside = preferred
      3. Inline signals: "must have X", "required: X", "X is mandatory"
      4. Everything else detected = general (treated as required by default)
    """
    text_lower = jd_text.lower()
    all_skills = set()
    required = set()
    preferred = set()

    # ── Step 1: section-based detection ──────────────────────────────────────
    REQUIRED_HEADERS = [
        "required skills", "must have", "mandatory skills", "must-have",
        "required qualifications", "requirements", "required experience",
        "minimum qualifications", "essential skills", "core requirements"
    ]
    PREFERRED_HEADERS = [
        "preferred skills", "nice to have", "good to have", "bonus",
        "preferred qualifications", "desirable skills", "advantageous",
        "plus", "would be a plus", "additional skills"
    ]

    def _skills_in_segment(segment: str) -> set:
        found = set()
        for skill in ALL_SKILLS:
            pat = r'(?<![a-zA-Z])' + re.escape(skill) + r'(?![a-zA-Z])'
            if re.search(pat, segment):
                found.add(skill)
        return found

    def _get_section_text(text: str, headers: list, window: int = 800) -> str:
        for header in headers:
            pat = rf'(?:^|\n)\s*(?:[-*•]?\s*)?{re.escape(header)}\s*[:\-\n]'
            m = re.search(pat, text)
            if m:
                segment = text[m.end(): m.end() + window]
                # stop at next section
                stop = re.search(r'\n[A-Z][A-Z\s]{3,}\n|\n\n[A-Z]', segment)
                if stop:
                    segment = segment[:stop.start()]
                return segment
        return ""

    req_segment  = _get_section_text(text_lower, REQUIRED_HEADERS)
    pref_segment = _get_section_text(text_lower, PREFERRED_HEADERS)

    if req_segment:
        required |= _skills_in_segment(req_segment)
    if pref_segment:
        preferred |= _skills_in_segment(pref_segment)

    # ── Step 2: inline signal patterns ───────────────────────────────────────
    REQUIRED_SIGNALS = [
        r'must\s+have\s+([\w\s\+\#\.]+?)(?:\.|,|\n|and)',
        r'required[:\s]+([\w\s\+\#\.]+?)(?:\.|,|\n)',
        r'mandatory[:\s]+([\w\s\+\#\.]+?)(?:\.|,|\n)',
        r'([\w\s\+\#\.]+?)\s+is\s+(?:required|mandatory|essential|must)',
        r'expertise\s+in\s+([\w\s\+\#\.]+?)(?:\.|,|\n)',
    ]
    PREFERRED_SIGNALS = [
        r'preferred[:\s]+([\w\s\+\#\.]+?)(?:\.|,|\n)',
        r'nice\s+to\s+have[:\s]+([\w\s\+\#\.]+?)(?:\.|,|\n)',
        r'plus\s+if\s+([\w\s\+\#\.]+?)(?:\.|,|\n)',
        r'([\w\s\+\#\.]+?)\s+(?:is\s+)?a\s+(?:plus|bonus|advantage)',
        r'familiarity\s+with\s+([\w\s\+\#\.]+?)(?:\.|,|\n)',
    ]

    for pat in REQUIRED_SIGNALS:
        for m in re.finditer(pat, text_lower):
            frag = m.group(1)
            required |= _skills_in_segment(frag)

    for pat in PREFERRED_SIGNALS:
        for m in re.finditer(pat, text_lower):
            frag = m.group(1)
            preferred |= _skills_in_segment(frag)

    # ── Step 3: detect all skills anywhere in JD ─────────────────────────────
    all_skills = _skills_in_segment(text_lower)

    # ── Step 4: classify unassigned skills ───────────────────────────────────
    # Skills not in required or preferred sections → treat as required
    unassigned = all_skills - required - preferred
    if req_segment:
        # If we found a required section, unassigned goes to preferred
        preferred |= unassigned
    else:
        # No explicit required section → all detected = required
        required |= unassigned

    # Preferred should never overlap with required
    preferred -= required

    return {
        "required": sorted(required),
        "preferred": sorted(preferred),
        "all": sorted(all_skills),
    }


# ── Keyword stuffing score ────────────────────────────────────────────────────
def compute_stuffing_penalty(text: str, jd_tokens: set) -> float:
    """
    Returns a penalty multiplier (0.7–1.0).
    If resume repeats JD keywords unnaturally many times → small penalty.
    """
    from collections import Counter
    words = re.findall(r'[a-z]+', text.lower())
    counts = Counter(words)
    overused = sum(1 for w in jd_tokens if counts.get(w, 0) > MAX_TOKEN_FREQUENCY * 3)
    if overused == 0:
        return 1.0
    # Max 30% penalty if very aggressive stuffing
    penalty = min(overused * 0.03, 0.30)
    return round(1.0 - penalty, 3)