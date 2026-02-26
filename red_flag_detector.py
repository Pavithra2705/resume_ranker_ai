"""
red_flag_detector.py — ResumeRank AI v2
Feature: Resume Red Flag Detector
Scans resumes for suspicious patterns that real recruiters look for.
"""

import re
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def detect_red_flags(resume_text: str) -> dict:
    """
    Scans a resume for red flags and positive signals.

    Returns:
        {
          "flags":        list of {type, severity, message, detail}
          "green_flags":  list of positive signals
          "risk_level":   "Low" / "Medium" / "High"
          "risk_score":   0-100 (higher = riskier)
          "summary":      one-line assessment
        }
    """
    flags       = []
    green_flags = []
    text_lower  = resume_text.lower()
    current_year = 2025

    # ── 1. Employment Gap Detection ───────────────────────────────────────────
    gaps = _detect_employment_gaps(resume_text, current_year)
    flags.extend(gaps)

    # ── 2. Job Hopping Detection ──────────────────────────────────────────────
    hopping = _detect_job_hopping(resume_text, current_year)
    flags.extend(hopping)

    # ── 3. Vague / Inflated Language ─────────────────────────────────────────
    vague = _detect_vague_language(text_lower)
    flags.extend(vague)

    # ── 4. Missing Critical Sections ─────────────────────────────────────────
    missing = _detect_missing_sections(text_lower)
    flags.extend(missing)

    # ── 5. Metric-Free Claims ─────────────────────────────────────────────────
    metric_flag = _detect_no_metrics(resume_text)
    if metric_flag:
        flags.append(metric_flag)

    # ── 6. Inconsistencies ────────────────────────────────────────────────────
    incon = _detect_inconsistencies(resume_text)
    flags.extend(incon)

    # ── Green Flags ───────────────────────────────────────────────────────────
    green_flags = _detect_green_flags(resume_text, text_lower)

    # ── Risk Score ────────────────────────────────────────────────────────────
    severity_weights = {"high": 25, "medium": 12, "low": 5}
    raw_risk = sum(severity_weights.get(f["severity"], 5) for f in flags)
    green_discount = len(green_flags) * 4
    risk_score = max(min(raw_risk - green_discount, 100), 0)

    if risk_score >= 50:
        risk_level = "High"
        summary = "Multiple red flags detected — recommend thorough screening before interview"
    elif risk_score >= 25:
        risk_level = "Medium"
        summary = "Some concerns noted — clarify during interview"
    else:
        risk_level = "Low"
        summary = "No major red flags — candidate profile appears consistent"

    return {
        "flags":       flags,
        "green_flags": green_flags,
        "risk_level":  risk_level,
        "risk_score":  risk_score,
        "summary":     summary,
        "flag_count":  len(flags),
        "green_count": len(green_flags),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_employment_gaps(text: str, current_year: int) -> list:
    """Detect gaps > 12 months between jobs."""
    flags = []

    # Extract all year mentions with context
    date_pat = re.compile(
        r'((?:19|20)\d{2})\s*[-–—to]+\s*((?:19|20)\d{2}|present|current|now)',
        re.IGNORECASE
    )
    ranges = []
    for m in date_pat.finditer(text.lower()):
        start = int(m.group(1))
        end_raw = m.group(2).strip()
        end = current_year if re.match(r'(?:present|current|now)', end_raw) else int(end_raw)
        if 1990 <= start <= current_year and start <= end:
            ranges.append((start, end))

    if len(ranges) < 2:
        return flags

    # Sort by start year
    ranges.sort(key=lambda x: x[0])

    # Check gaps between consecutive jobs
    for i in range(1, len(ranges)):
        prev_end   = ranges[i-1][1]
        curr_start = ranges[i][0]
        gap = curr_start - prev_end

        if gap >= 2:
            flags.append({
                "type":     "Employment Gap",
                "severity": "high",
                "message":  f"Gap of ~{gap} year(s) detected between {prev_end} and {curr_start}",
                "detail":   "Ask candidate to explain this gap during interview",
                "icon":     "⏳"
            })
        elif gap == 1:
            flags.append({
                "type":     "Employment Gap",
                "severity": "medium",
                "message":  f"Possible 1-year gap between {prev_end} and {curr_start}",
                "detail":   "Minor — worth a brief clarification",
                "icon":     "⏳"
            })

    return flags


def _detect_job_hopping(text: str, current_year: int) -> list:
    """Detect if candidate changed jobs too frequently (3+ jobs in 2 years)."""
    flags = []

    date_pat = re.compile(
        r'((?:19|20)\d{2})\s*[-–—to]+\s*((?:19|20)\d{2}|present|current|now)',
        re.IGNORECASE
    )
    ranges = []
    for m in date_pat.finditer(text.lower()):
        start = int(m.group(1))
        end_raw = m.group(2).strip()
        end = current_year if re.match(r'(?:present|current|now)', end_raw) else int(end_raw)
        if 1990 <= start <= current_year and start <= end:
            ranges.append((start, end))

    if len(ranges) < 2:
        return flags

    ranges.sort(key=lambda x: x[0])

    # Count short tenures (< 1 year)
    short_tenures = [(s, e) for s, e in ranges if (e - s) < 1]
    very_short    = [(s, e) for s, e in ranges if (e - s) == 0]

    if len(very_short) >= 3:
        flags.append({
            "type":     "Job Hopping",
            "severity": "high",
            "message":  f"{len(very_short)} positions with less than 1 year tenure detected",
            "detail":   "High turnover risk — explore reasons for frequent changes",
            "icon":     "🔄"
        })
    elif len(short_tenures) >= 2:
        flags.append({
            "type":     "Job Hopping",
            "severity": "medium",
            "message":  f"{len(short_tenures)} short-tenure positions (< 1 year) detected",
            "detail":   "Ask about reasons for leaving each role",
            "icon":     "🔄"
        })

    return flags


def _detect_vague_language(text_lower: str) -> list:
    """Detect inflated titles and vague buzzwords."""
    flags = []

    INFLATED_TITLES = [
        (r'\bceo\s+of\s+(?:self|my|own|freelance)\b', "Self-declared CEO title", "high"),
        (r'\bfounder\b.*\b(?:self|solo|single)\b', "Solo founder claim without team context", "medium"),
        (r'\bexpert\s+in\s+everything\b', "Unrealistic 'expert in everything' claim", "high"),
        (r'\b(?:10|15|20)\+\s*years.*(?:python|react|kubernetes|docker)\b',
         "Years exceed technology's existence", "high"),
    ]

    BUZZWORD_OVERUSE = [
        r'\bsynerg', r'\bparadigm\b', r'\bleverage[sd]?\b',
        r'\bthought\s+leader\b', r'\bvisionary\b', r'\bgame.?changer\b',
        r'\bdisruptive\b', r'\binnovative\s+solutions\b',
    ]

    for pat, msg, severity in INFLATED_TITLES:
        if re.search(pat, text_lower):
            flags.append({
                "type":     "Inflated Language",
                "severity": severity,
                "message":  msg,
                "detail":   "Verify claims with references or portfolio",
                "icon":     "⚠️"
            })

    buzz_count = sum(1 for p in BUZZWORD_OVERUSE if re.search(p, text_lower))
    if buzz_count >= 4:
        flags.append({
            "type":     "Buzzword Heavy",
            "severity": "low",
            "message":  f"{buzz_count} corporate buzzwords detected — low substance signals",
            "detail":   "Ask for specific examples and quantified achievements",
            "icon":     "💬"
        })

    return flags


def _detect_missing_sections(text_lower: str) -> list:
    """Flag missing critical resume sections."""
    flags = []

    REQUIRED_SECTIONS = {
        "contact":    (r'(?:email|phone|mobile|linkedin|github|contact)', "Contact information"),
        "experience": (r'(?:experience|employment|work history|worked at|position)', "Work experience"),
        "education":  (r'(?:education|degree|university|college|bachelor|master|phd)', "Education"),
        "skills":     (r'(?:skills|technologies|tools|languages|frameworks)', "Skills section"),
    }

    for key, (pat, label) in REQUIRED_SECTIONS.items():
        if not re.search(pat, text_lower):
            severity = "high" if key in ["experience", "education"] else "medium"
            flags.append({
                "type":     "Missing Section",
                "severity": severity,
                "message":  f"No {label} section detected",
                "detail":   f"Resume may be incomplete or improperly formatted",
                "icon":     "📋"
            })

    return flags


def _detect_no_metrics(resume_text: str) -> dict | None:
    """Flag resumes with zero quantified achievements."""
    METRIC_PATTERNS = [
        r'\d+\s*%',                    # percentages
        r'\$[\d,]+',                   # dollar amounts
        r'₹[\d,]+',                   # rupee amounts
        r'\b\d+[xX]\b',               # multiples (3x, 10x)
        r'\bteam\s+of\s+\d+',         # team size
        r'\b\d+\s*(?:users?|customers?|clients?)\b',
        r'\breduced\s+by\s+\d+',
        r'\bincreased\s+by\s+\d+',
        r'\bimproved\s+by\s+\d+',
        r'\b\d+\s*(?:lakh|crore|million|billion|k\b)',
    ]
    matches = sum(1 for p in METRIC_PATTERNS if re.search(p, resume_text, re.IGNORECASE))
    if matches == 0:
        return {
            "type":     "No Metrics",
            "severity": "medium",
            "message":  "Zero quantified achievements detected",
            "detail":   "Strong resumes have numbers — ask for specific impact examples",
            "icon":     "📊"
        }
    return None


def _detect_inconsistencies(resume_text: str) -> list:
    """Detect date inconsistencies and other red flags."""
    flags = []
    current_year = 2025

    # Future dates
    future_pat = re.compile(r'(20(?:2[6-9]|[3-9]\d))\s*[-–—to]', re.IGNORECASE)
    for m in future_pat.finditer(resume_text):
        yr = int(m.group(1))
        if yr > current_year:
            flags.append({
                "type":     "Date Inconsistency",
                "severity": "high",
                "message":  f"Future date {yr} found in work history",
                "detail":   "Verify timeline — may indicate data entry error or fabrication",
                "icon":     "📅"
            })

    # Very old dates mixed with recent ones (possible fabricated experience)
    years_found = [int(y) for y in re.findall(r'\b((?:19|20)\d{2})\b', resume_text)
                   if 1970 <= int(y) <= current_year]
    if years_found:
        span = max(years_found) - min(years_found)
        if span > 35:
            flags.append({
                "type":     "Unusual Timeline",
                "severity": "low",
                "message":  f"Resume spans {span} years — unusually wide date range",
                "detail":   "Verify all dates and roles are accurately represented",
                "icon":     "📅"
            })

    return flags


def _detect_green_flags(resume_text: str, text_lower: str) -> list:
    """Identify positive signals that boost candidate credibility."""
    green_flags = []

    checks = [
        (r'github\.com|gitlab\.com|bitbucket',           "Active code repository (GitHub/GitLab)"),
        (r'linkedin\.com',                                "LinkedIn profile present"),
        (r'published|publication|journal|conference|paper',"Research publications"),
        (r'award|winner|prize|recognition|honour|honor',  "Awards or recognition"),
        (r'certified|certification|certificate',          "Professional certifications"),
        (r'led\s+(?:a\s+)?team|managed\s+(?:a\s+)?team|team\s+lead|team\s+of\s+\d+',
                                                          "Demonstrated leadership"),
        (r'open\s*source|contributed\s+to|contributor',  "Open source contributions"),
        (r'\d+\s*%|\d+[xX]|₹[\d,]+|\$[\d,]+',           "Quantified achievements"),
        (r'promoted|promotion',                           "Career progression / promotion"),
        (r'patent',                                       "Patent filed/granted"),
        (r'keynote|speaker|talk\s+at|presented\s+at',    "Public speaking / conference talk"),
    ]

    for pat, label in checks:
        if re.search(pat, text_lower):
            green_flags.append(label)

    return green_flags
