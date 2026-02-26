"""
scoring.py — ResumeRank AI v2
Core scoring engine. Fits TF-IDF ONCE on all documents for comparable scores.
"""

import math
import re
from config import WEIGHT_PROFILES, GRADE_SCALE, SKILL_WEIGHTS, DEFAULT_PROFILE
from nlp_utils import (
    preprocess_text, extract_skills, extract_years_of_experience,
    extract_education_score, compute_stuffing_penalty, extract_required_skills
)

# ── Optional sklearn ──────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def rank_all_resumes(job_description: str, resumes: list[dict], profile: str = DEFAULT_PROFILE) -> list[dict]:
    """
    Score and rank all resumes together.

    Args:
        job_description: Raw JD text
        resumes: list of {"filename": str, "text": str}
        profile: weight profile key from config.WEIGHT_PROFILES

    Returns:
        Sorted list of result dicts, rank 1 = best match
    """
    weights = WEIGHT_PROFILES.get(profile, WEIGHT_PROFILES[DEFAULT_PROFILE])["weights"]

    # ── Preprocess all texts ONCE ────────────────────────────────────────────
    processed_jd = preprocess_text(job_description)
    processed_resumes = [preprocess_text(r["text"]) for r in resumes]

    # ── Fit TF-IDF on ALL documents (JD + resumes) ───────────────────────────
    tfidf_scores = _batch_tfidf_similarity(processed_jd, processed_resumes)

    # ── Extract JD skills, keywords, and required/preferred split ────────────
    jd_skills_data    = extract_skills(job_description)
    jd_required_data  = extract_required_skills(job_description)
    jd_skills         = set(jd_skills_data["all"])
    jd_required_skills= set(jd_required_data["required"])
    jd_preferred_skills=set(jd_required_data["preferred"])
    jd_words          = set(processed_jd.split())

    # ── Extract experience requirement from JD ────────────────────────────────
    jd_exp_req = _extract_jd_experience_requirement(job_description)

    # ── Extract education requirement from JD ────────────────────────────────
    jd_edu_req = _extract_jd_education_requirement(job_description)

    # ── Score each resume ────────────────────────────────────────────────────
    results = []
    for i, resume in enumerate(resumes):
        raw_text = resume["text"]
        proc_text = processed_resumes[i]

        result = _score_single(
            filename=resume["filename"],
            raw_text=raw_text,
            proc_text=proc_text,
            tfidf_score=tfidf_scores[i],
            jd_words=jd_words,
            jd_skills=jd_skills,
            jd_required_skills=jd_required_skills,
            jd_preferred_skills=jd_preferred_skills,
            jd_exp_req=jd_exp_req,
            jd_edu_req=jd_edu_req,
            weights=weights,
        )
        results.append(result)

    # ── Sort & assign ranks ───────────────────────────────────────────────────
    results.sort(key=lambda x: x["final_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BATCH TF-IDF — fit once, score all
# ─────────────────────────────────────────────────────────────────────────────
def _batch_tfidf_similarity(processed_jd: str, processed_resumes: list[str]) -> list[float]:
    """
    Fit TF-IDF on [JD + all resumes], then return cosine similarity of each
    resume against JD. Scores are globally comparable.
    """
    if not SKLEARN_OK:
        return [_jaccard(processed_jd, r) for r in processed_resumes]

    corpus = [processed_jd] + processed_resumes
    try:
        vectorizer = TfidfVectorizer(
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
            sublinear_tf=True       # log normalization reduces TF inflation
        )
        matrix = vectorizer.fit_transform(corpus)
        jd_vec = matrix[0]
        resume_vecs = matrix[1:]
        sims = cosine_similarity(jd_vec, resume_vecs)[0]
        return [float(s) for s in sims]
    except Exception:
        return [_jaccard(processed_jd, r) for r in processed_resumes]


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Extract experience requirement from JD text
# ─────────────────────────────────────────────────────────────────────────────
def _extract_jd_experience_requirement(jd_text: str) -> dict:
    """
    Returns {"min": float, "max": float} years required from JD.
    E.g. "4-7 years" → {"min": 4, "max": 7}
         "5+ years"  → {"min": 5, "max": 99}
         "5 years"   → {"min": 5, "max": 99}
    """
    text = jd_text.lower()
    # Range: "4-7 years", "4 to 7 years"
    range_pat = re.compile(r'(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)')
    m = range_pat.search(text)
    if m:
        return {"min": float(m.group(1)), "max": float(m.group(2))}
    # Minimum: "5+ years", "minimum 5 years", "at least 5 years"
    min_pat = re.compile(r'(?:minimum|at\s+least|min\.?\s*)?(\d+(?:\.\d+)?)\+\s*(?:years?|yrs?)')
    m = min_pat.search(text)
    if m:
        return {"min": float(m.group(1)), "max": 99}
    # Plain: "5 years of experience"
    plain_pat = re.compile(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)')
    m = plain_pat.search(text)
    if m:
        val = float(m.group(1))
        return {"min": max(val - 1, 0), "max": val + 3}
    return {"min": 0, "max": 99}


def _extract_jd_education_requirement(jd_text: str) -> str:
    """Returns the minimum education level the JD expects: 'bachelor', 'master', 'phd', or 'any'."""
    text = jd_text.lower()
    if re.search(r'\bph\.?d\b|\bdoctorate\b', text):
        return 'phd'
    if re.search(r'\bmaster\b|\bm\.?tech\b|\bmba\b|\bm\.?sc\b|\bm\.?s\b', text):
        return 'master'
    if re.search(r'\bbachelor\b|\bb\.?tech\b|\bb\.?e\b|\bb\.?sc\b', text):
        return 'bachelor'
    return 'any'


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE RESUME SCORER
# ─────────────────────────────────────────────────────────────────────────────
def _score_single(filename, raw_text, proc_text, tfidf_score,
                  jd_words, jd_skills, jd_required_skills, jd_preferred_skills,
                  jd_exp_req, jd_edu_req, weights) -> dict:

    scores = {}

    # ── 1. TF-IDF Similarity ─────────────────────────────────────────────────
    scores["tfidf_similarity"] = tfidf_score

    # ── 2. Keyword Match (sqrt-normalized to reduce long-JD bias) ────────────
    resume_words = set(proc_text.split())
    matched_kw = jd_words & resume_words
    if jd_words:
        raw_ratio = len(matched_kw) / math.sqrt(len(jd_words))
        kw_score = min(raw_ratio, 1.0)
    else:
        kw_score = 0.0
    scores["keyword_match"] = kw_score
    scores["matched_keywords"] = sorted(matched_kw)[:12]

    # ── 3. Skill Overlap (weighted by tier) ──────────────────────────────────
    resume_skills_data = extract_skills(raw_text)
    resume_skills  = set(resume_skills_data["all"])
    section_skills = set(resume_skills_data["section_skills"])

    if jd_skills:
        overlap = jd_skills & resume_skills
        weighted_overlap = sum(SKILL_WEIGHTS.get(s, 1.0) for s in overlap)
        weighted_jd      = sum(SKILL_WEIGHTS.get(s, 1.0) for s in jd_skills)
        base_skill_score = weighted_overlap / weighted_jd if weighted_jd else 0.0

        # Section-aware bonus
        section_bonus = 0.0
        if section_skills:
            section_overlap = jd_skills & section_skills
            if section_overlap:
                section_bonus = 0.10 * (len(section_overlap) / len(jd_skills))

        skill_score = min(base_skill_score + section_bonus, 1.0)
    else:
        skill_score = 0.0
        overlap = set()

    scores["skill_overlap"]    = skill_score
    scores["skills_found"]     = sorted(resume_skills)[:15]
    scores["skills_matched"]   = sorted(overlap)[:12]
    scores["skills_missing"]   = sorted(jd_skills - resume_skills)[:10]

    # ── 4. Experience ─────────────────────────────────────────────────────────
    years = extract_years_of_experience(raw_text)
    exp_score = min(years / 10.0, 1.0)
    scores["experience"]      = exp_score
    scores["years_experience"]= years

    # ── 5. Education ─────────────────────────────────────────────────────────
    edu_score, edu_label = extract_education_score(raw_text)
    scores["education"]      = edu_score
    scores["education_label"]= edu_label

    # ── 6. Keyword Stuffing Penalty ───────────────────────────────────────────
    stuffing_multiplier = compute_stuffing_penalty(raw_text, jd_words)
    scores["stuffing_multiplier"] = stuffing_multiplier

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 1: Required Skill Boost / Penalty
    # ─────────────────────────────────────────────────────────────────────────
    required_boost = 1.0
    required_penalty = 1.0
    required_hit  = set()
    required_miss = set()

    if jd_required_skills:
        required_hit  = jd_required_skills & resume_skills
        required_miss = jd_required_skills - resume_skills
        coverage = len(required_hit) / len(jd_required_skills)

        if coverage == 1.0:
            # All required skills present → +8% boost
            required_boost = 1.08
        elif coverage >= 0.80:
            # 80%+ present → small boost
            required_boost = 1.04
        elif coverage >= 0.60:
            # 60-80% → neutral
            required_boost = 1.0
        elif coverage >= 0.40:
            # 40-60% → mild penalty
            required_penalty = 0.92
        else:
            # <40% of required skills present → heavy penalty
            required_penalty = 0.80

    # Preferred skill bonus (smaller, additive)
    preferred_hit = set()
    preferred_bonus = 0.0
    if jd_preferred_skills:
        preferred_hit = jd_preferred_skills & resume_skills
        if preferred_hit:
            preferred_bonus = 0.03 * min(len(preferred_hit) / max(len(jd_preferred_skills), 1), 1.0)

    scores["required_boost"]    = round(required_boost, 3)
    scores["required_penalty"]  = round(required_penalty, 3)
    scores["required_hit"]      = sorted(required_hit)
    scores["required_miss"]     = sorted(required_miss)
    scores["preferred_hit"]     = sorted(preferred_hit)
    scores["required_coverage"] = round(len(required_hit) / len(jd_required_skills) * 100, 1) if jd_required_skills else 100.0

    # ─────────────────────────────────────────────────────────────────────────
    # Compute raw final score (before length and ATS adjustments)
    # ─────────────────────────────────────────────────────────────────────────
    raw_final = (
        weights["tfidf_similarity"] * scores["tfidf_similarity"] +
        weights["keyword_match"]    * scores["keyword_match"] +
        weights["skill_overlap"]    * scores["skill_overlap"] +
        weights["experience"]       * scores["experience"] +
        weights["education"]        * scores["education"]
    )

    # Apply stuffing + required skill multipliers
    raw_final = raw_final * stuffing_multiplier * required_penalty * required_boost
    # Add preferred skill bonus (flat additive, small)
    raw_final = min(raw_final + preferred_bonus, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 2: Resume Length Normalization
    # ─────────────────────────────────────────────────────────────────────────
    word_count = len(proc_text.split())
    length_multiplier = 1.0
    if word_count < 50:
        length_multiplier = 0.75   # extremely sparse — likely garbled/scanned
    elif word_count < 100:
        length_multiplier = 0.85   # very short resume
    elif word_count < 150:
        length_multiplier = 0.93   # short but passable
    # 150+ words → no penalty
    raw_final *= length_multiplier

    scores["word_count"]         = word_count
    scores["length_multiplier"]  = length_multiplier

    final = round(min(raw_final * 100, 100.0), 2)
    scores["final_score"] = final

    # ── Grade ─────────────────────────────────────────────────────────────────
    grade, label = _get_grade(final)
    scores["grade"]        = grade
    scores["label"]        = label
    scores["filename"]     = filename
    scores["text_preview"] = raw_text[:300].strip() + ("..." if len(raw_text) > 300 else "")

    # Component percentages for UI bars
    scores["tfidf_pct"]      = round(scores["tfidf_similarity"] * 100, 1)
    scores["keyword_pct"]    = round(scores["keyword_match"] * 100, 1)
    scores["skill_pct"]      = round(scores["skill_overlap"] * 100, 1)
    scores["experience_pct"] = round(scores["experience"] * 100, 1)
    scores["education_pct"]  = round(scores["education"] * 100, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 3: Explainability
    # ─────────────────────────────────────────────────────────────────────────
    scores["explanation"] = _generate_explanation(scores, jd_exp_req, jd_edu_req, jd_required_skills)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def _generate_explanation(scores: dict, jd_exp_req: dict, jd_edu_req: str,
                           jd_required_skills: set) -> list:
    """
    Returns a list of plain-English explanation strings describing
    why this candidate received their score. Each point maps to one signal.
    """
    lines = []
    years      = scores.get("years_experience", 0)
    edu_label  = scores.get("education_label", "Not specified")
    req_hit    = scores.get("required_hit", [])
    req_miss   = scores.get("required_miss", [])
    pref_hit   = scores.get("preferred_hit", [])
    req_cov    = scores.get("required_coverage", 0)
    word_count = scores.get("word_count", 0)
    stuffing   = scores.get("stuffing_multiplier", 1.0)
    tfidf      = scores.get("tfidf_pct", 0)
    kw_pct     = scores.get("keyword_pct", 0)
    skill_pct  = scores.get("skill_pct", 0)

    # ── Semantic similarity ───────────────────────────────────────────────────
    if tfidf >= 70:
        lines.append(f"✅ Excellent semantic match with JD — TF-IDF similarity {tfidf}%")
    elif tfidf >= 45:
        lines.append(f"🟡 Moderate semantic overlap with JD — TF-IDF similarity {tfidf}%")
    else:
        lines.append(f"❌ Low semantic similarity to JD — TF-IDF score only {tfidf}%")

    # ── Required skills ───────────────────────────────────────────────────────
    if jd_required_skills:
        if req_cov == 100.0:
            lines.append(f"✅ All {len(jd_required_skills)} required skills present — full ATS compliance (+8% boost applied)")
        elif req_cov >= 80:
            lines.append(f"✅ {req_cov}% required skill coverage ({len(req_hit)}/{len(jd_required_skills)}) — strong match (+4% boost)")
        elif req_cov >= 60:
            lines.append(f"🟡 {req_cov}% required skill coverage — {', '.join(req_miss[:3]) if req_miss else 'some gaps'} missing")
        elif req_cov >= 40:
            lines.append(f"🟠 Only {req_cov}% required skills matched — missing: {', '.join(req_miss[:4])} (−8% penalty)")
        else:
            names = ', '.join(req_miss[:5])
            lines.append(f"❌ Critical skill gap — only {req_cov}% required skills found. Missing: {names} (−20% penalty)")
    else:
        if skill_pct >= 70:
            lines.append(f"✅ Strong skill match — {skill_pct}% of JD skills detected in resume")
        elif skill_pct >= 45:
            lines.append(f"🟡 Partial skill match — {skill_pct}% of JD skills detected")
        else:
            lines.append(f"❌ Weak skill match — only {skill_pct}% of JD skills found")

    # ── Preferred skills ─────────────────────────────────────────────────────
    if pref_hit:
        lines.append(f"⭐ Bonus: has {len(pref_hit)} preferred skill(s) — {', '.join(pref_hit[:4])}")

    # ── Experience ───────────────────────────────────────────────────────────
    exp_min = jd_exp_req.get("min", 0)
    exp_max = jd_exp_req.get("max", 99)
    if years == 0:
        lines.append("⚠️  No experience duration detected in resume")
    elif years >= exp_min and years <= exp_max:
        lines.append(f"✅ Experience ({years} yrs) is within the required range ({exp_min}–{str(exp_max) if exp_max < 99 else str(exp_min) + '+'} yrs)")
    elif years > exp_max and exp_max < 99:
        lines.append(f"🟡 Overqualified — {years} yrs experience exceeds max required {exp_max} yrs")
    elif years < exp_min:
        lines.append(f"🟠 Under-experienced — {years} yrs found, JD requires minimum {exp_min} yrs")

    # ── Education ─────────────────────────────────────────────────────────────
    EDU_RANK = {"phd": 4, "master": 3, "bachelor": 2, "diploma": 1, "any": 0}
    edu_label_lower = edu_label.lower()
    candidate_rank = 0
    for key in EDU_RANK:
        if key in edu_label_lower:
            candidate_rank = EDU_RANK[key]
            break
    required_rank = EDU_RANK.get(jd_edu_req, 0)

    if edu_label == "Not specified":
        lines.append("⚠️  No education qualification detected in resume")
    elif candidate_rank >= required_rank + 1:
        lines.append(f"✅ Education ({edu_label}) exceeds JD requirement ({jd_edu_req.title()})")
    elif candidate_rank == required_rank:
        lines.append(f"✅ Education ({edu_label}) meets JD requirement ({jd_edu_req.title()})")
    else:
        lines.append(f"🟠 Education ({edu_label}) is below JD requirement ({jd_edu_req.title()})")

    # ── Resume quality signals ────────────────────────────────────────────────
    if word_count < 100:
        lines.append(f"⚠️  Resume is very short ({word_count} words) — length penalty applied (×{scores.get('length_multiplier',1.0)})")
    elif word_count > 400:
        lines.append(f"✅ Detailed resume ({word_count} words) — good coverage of experience")
    else:
        lines.append(f"ℹ️  Resume length is adequate ({word_count} words)")

    if stuffing < 1.0:
        lines.append(f"⚠️  Keyword stuffing detected — score penalised (×{stuffing})")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
def _get_grade(score: float) -> tuple:
    for threshold, grade, label in GRADE_SCALE:
        if score >= threshold:
            return grade, label
    return "F", "Poor Match"