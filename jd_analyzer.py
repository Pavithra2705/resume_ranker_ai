"""
jd_analyzer.py — ResumeRank AI v2
Feature: JD Quality Scorer + Bias Detector
Analyzes a job description for clarity, completeness, and bias.
"""

import re
from config import ALL_SKILLS, TIER1_SKILLS

# ── Biased language patterns ──────────────────────────────────────────────────
BIAS_PATTERNS = {
    "age_bias": {
        "patterns": [
            r'\byoung\b', r'\bdigital native\b', r'\brecent graduate only\b',
            r'\bfresh\s+out\b', r'\bage\s+\d+\b', r'\bunder\s+\d+\s+years\s+old\b',
            r'\bnative\s+speaker\b', r'\benergetic\s+young\b', r'\bjunior\s+minded\b',
        ],
        "label": "Age Bias",
        "suggestion": "Avoid age-specific language. Focus on skills and experience level instead.",
        "severity": "high"
    },
    "gender_bias": {
        "patterns": [
            r'\brockstar\b', r'\bninja\b', r'\bguru\b', r'\bwizard\b',
            r'\bhe\s+will\b', r'\bshe\s+will\b', r'\bhis\s+duties\b',
            r'\bmanpower\b', r'\bchairman\b', r'\bsalesman\b',
            r'\bdominant\b', r'\baggressive\b', r'\bcompetitive\s+individual\b',
        ],
        "label": "Gender Bias",
        "suggestion": "Use gender-neutral language. Replace 'rockstar/ninja' with specific skill requirements.",
        "severity": "high"
    },
    "cultural_bias": {
        "patterns": [
            r'\bnative\s+english\b', r'\bmother\s+tongue\b', r'\bperfect\s+english\b',
            r'\bcultural\s+fit\b', r'\bjust\s+like\s+us\b', r'\bgood\s+personality\b',
        ],
        "label": "Cultural Bias",
        "suggestion": "Specify required communication skills objectively (e.g., 'strong written and verbal communication').",
        "severity": "medium"
    },
    "vague_requirements": {
        "patterns": [
            r'\bpassionate\b', r'\bself.?starter\b', r'\bteam\s+player\b',
            r'\bgo.?getter\b', r'\bthink\s+outside\s+the\s+box\b',
            r'\bwear\s+many\s+hats\b', r'\bfast.?paced\b', r'\bfast\s+learner\b',
            r'\bdynamic\b', r'\bsynergy\b', r'\bpivot\b',
        ],
        "label": "Vague Language",
        "suggestion": "Replace vague buzzwords with specific, measurable requirements.",
        "severity": "low"
    }
}

# ── Good JD patterns ──────────────────────────────────────────────────────────
GOOD_SECTIONS = [
    "responsibilities", "requirements", "qualifications", "skills",
    "about", "benefits", "compensation", "experience", "education"
]

SALARY_PATTERNS = [
    r'\bsalary\b', r'\bctc\b', r'\blpa\b', r'\bper\s+annum\b',
    r'\bcompensation\b', r'\bstipend\b', r'\bpay\b', r'\bremuneration\b'
]

METRIC_PATTERNS = [
    r'\d+\+?\s*years?', r'\d+\s*%', r'\$\d+', r'₹\d+',
    r'\d+\s*(?:lpa|ctc|k\b)', r'\bteam\s+of\s+\d+\b'
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def analyze_jd(jd_text: str) -> dict:
    """
    Analyze a job description for quality, completeness, and bias.

    Returns a dict with:
      - overall_score (0-100)
      - grade (A/B/C/D/F)
      - dimensions: {clarity, completeness, skill_specificity, bias_free}
      - bias_flags: list of detected bias issues
      - suggestions: list of improvement tips
      - stats: word count, skill count, section count
      - verdict: one-line summary
    """
    text_lower = jd_text.lower()
    word_count = len(jd_text.split())
    suggestions = []
    bias_flags  = []

    # ── 1. Completeness (25 pts) ──────────────────────────────────────────────
    sections_found = [s for s in GOOD_SECTIONS if s in text_lower]
    completeness_score = min(len(sections_found) / len(GOOD_SECTIONS) * 100, 100)

    has_salary   = any(re.search(p, text_lower) for p in SALARY_PATTERNS)
    has_metrics  = len(re.findall('|'.join(METRIC_PATTERNS), text_lower)) >= 2
    has_edu_req  = bool(re.search(r'\b(?:bachelor|master|degree|phd|b\.tech|m\.tech)\b', text_lower))
    has_exp_req  = bool(re.search(r'\d+\+?\s*years?\s*(?:of\s+)?(?:experience|exp)', text_lower))

    if not has_salary:
        suggestions.append("💰 Add salary/compensation range — improves candidate quality by 40%")
    if not has_edu_req:
        suggestions.append("🎓 Specify minimum education requirement (Bachelor's/Master's etc.)")
    if not has_exp_req:
        suggestions.append("📅 Specify years of experience required (e.g., '3–5 years')")
    if not has_metrics:
        suggestions.append("📊 Add measurable expectations (e.g., team size, revenue scale, SLA targets)")

    completeness_bonus = (has_salary + has_metrics + has_edu_req + has_exp_req) * 5
    completeness_final = min(completeness_score * 0.6 + completeness_bonus * 4, 100)

    # ── 2. Clarity / Length (25 pts) ─────────────────────────────────────────
    if word_count < 100:
        clarity_score = 20
        suggestions.append("📝 JD is too short (under 100 words) — candidates need more context")
    elif word_count < 200:
        clarity_score = 50
        suggestions.append("📝 JD is brief — consider expanding responsibilities and requirements sections")
    elif word_count <= 600:
        clarity_score = 100   # ideal range
    elif word_count <= 900:
        clarity_score = 80
        suggestions.append("✂️  JD is getting long — consider trimming to under 600 words for better readability")
    else:
        clarity_score = 55
        suggestions.append("✂️  JD is too long (900+ words) — candidates may lose interest. Aim for 400–600 words")

    # Sentence structure check — too many bullet points with no context
    sentences = [s.strip() for s in re.split(r'[.!?]', jd_text) if len(s.strip()) > 20]
    avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    if avg_sentence_len < 5:
        clarity_score = max(clarity_score - 15, 20)
        suggestions.append("✍️  Sentences are very short/fragmented — add context to bullet points")

    # ── 3. Skill Specificity (25 pts) ─────────────────────────────────────────
    skills_in_jd = []
    for skill in ALL_SKILLS:
        pat = r'(?<![a-zA-Z])' + re.escape(skill) + r'(?![a-zA-Z])'
        if re.search(pat, text_lower):
            skills_in_jd.append(skill)

    tier1_in_jd = [s for s in skills_in_jd if s in TIER1_SKILLS]
    skill_count  = len(skills_in_jd)
    t1_count     = len(tier1_in_jd)

    if skill_count == 0:
        skill_score = 10
        suggestions.append("🔧 No specific technical skills mentioned — add required tools/technologies")
    elif skill_count < 3:
        skill_score = 35
        suggestions.append(f"🔧 Only {skill_count} skill(s) detected — list all required technologies clearly")
    elif skill_count < 6:
        skill_score = 65
    elif skill_count <= 15:
        skill_score = 100
    else:
        skill_score = 75
        suggestions.append(f"⚠️  {skill_count} skills listed — this may overwhelm candidates. Prioritize must-haves vs. nice-to-haves")

    # Bonus for having required vs preferred sections
    has_req_section  = bool(re.search(r'required\s+skills?|must\s+have|mandatory', text_lower))
    has_pref_section = bool(re.search(r'preferred\s+skills?|nice\s+to\s+have|good\s+to\s+have|plus', text_lower))
    if not has_req_section:
        suggestions.append("🎯 Separate 'Required Skills' from 'Preferred Skills' — makes ATS scoring more accurate")
    if has_req_section and has_pref_section:
        skill_score = min(skill_score + 10, 100)

    # ── 4. Bias Detection (25 pts) ────────────────────────────────────────────
    bias_deductions = 0
    for bias_type, config in BIAS_PATTERNS.items():
        matches = []
        for pat in config["patterns"]:
            found = re.findall(pat, text_lower)
            matches.extend(found)
        if matches:
            deduct = 20 if config["severity"] == "high" else 10 if config["severity"] == "medium" else 5
            bias_deductions += deduct
            bias_flags.append({
                "type":       config["label"],
                "severity":   config["severity"],
                "matches":    list(set(matches))[:4],
                "suggestion": config["suggestion"]
            })

    bias_score = max(100 - bias_deductions, 0)

    # ── Overall Score ─────────────────────────────────────────────────────────
    overall = round(
        completeness_final * 0.25 +
        clarity_score      * 0.25 +
        skill_score        * 0.25 +
        bias_score         * 0.25,
        1
    )

    # ── Grade ─────────────────────────────────────────────────────────────────
    if overall >= 85:
        grade, verdict = "A", "Excellent JD — well-structured, specific, and bias-free"
    elif overall >= 70:
        grade, verdict = "B", "Good JD — minor improvements will significantly boost response rate"
    elif overall >= 55:
        grade, verdict = "C", "Average JD — missing key sections and specificity"
    elif overall >= 40:
        grade, verdict = "D", "Weak JD — vague requirements will attract poor-fit candidates"
    else:
        grade, verdict = "F", "Poor JD — major revision needed before posting"

    # ── Priority suggestions (top 4 only) ────────────────────────────────────
    top_suggestions = suggestions[:4]

    return {
        "overall_score":    overall,
        "grade":            grade,
        "verdict":          verdict,
        "dimensions": {
            "completeness":      round(completeness_final, 1),
            "clarity":           round(clarity_score, 1),
            "skill_specificity": round(skill_score, 1),
            "bias_free":         round(bias_score, 1),
        },
        "bias_flags":    bias_flags,
        "suggestions":   top_suggestions,
        "stats": {
            "word_count":       word_count,
            "skill_count":      skill_count,
            "tier1_skills":     t1_count,
            "sections_found":   sections_found,
            "has_salary":       has_salary,
            "has_exp_req":      has_exp_req,
            "has_edu_req":      has_edu_req,
            "skills_detected":  skills_in_jd[:12],
        }
    }
