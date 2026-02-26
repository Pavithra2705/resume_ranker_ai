# ResumeRank AI v3
### Intelligent Applicant Tracking & Resume Ranking System

> Rank smarter. Hire faster. — An industry-grade ATS built with Python, Flask, and NLP.

---

## What It Does

ResumeRank AI automatically scores and ranks candidate resumes against a Job Description using Natural Language Processing and Machine Learning. Unlike basic keyword matchers, it fits a **global TF-IDF model** across all resumes simultaneously — so every candidate gets a fair, comparable score.

Upload resumes in PDF, TXT, or DOCX format, paste a job description, and get ranked results with letter grades, skill gap analysis, radar charts, and plain-English explanations — all in seconds.

---

## Features

### Core Scoring Engine
| Signal | Method |
|--------|--------|
| Semantic Similarity | Global TF-IDF + Cosine Similarity (all docs fitted together) |
| Keyword Match | √-normalised overlap (prevents long-JD bias) |
| Skill Overlap | 3-tier weighted taxonomy — 80+ skills across Tier 1/2/3 |
| Experience | Date-range regex extraction (e.g. `2019–Present`) |
| Education | Word-boundary regex — no false positives |

### v3 Advanced Features

**🔍 JD Quality Scorer**
Grades the job description itself before you rank resumes. Scores on 4 dimensions:
- Completeness (salary, exp req, edu req, responsibilities)
- Clarity (word count, sentence structure)
- Skill Specificity (required vs. preferred skills split)
- Bias-Free (detects age, gender, cultural bias patterns)

**🚩 Red Flag Detector**
Scans each resume for recruiter red flags:
- Employment gaps > 1 year
- Job hopping (3+ short tenures)
- Inflated titles / buzzword overuse
- Missing critical sections
- Zero quantified achievements

Also detects ✅ **Green Flags** — GitHub profile, certifications, promotions, publications, awards.

**⚖️ Candidate Comparison Mode**
Select any 2 candidates for a side-by-side head-to-head breakdown:
- Visual signal bars for all 5 scoring dimensions
- Skills unique to each candidate vs. common skills
- Auto-generated hiring recommendation

### Additional Capabilities
- **Required Skill ATS Boost** — ×1.08 boost for 100% required skill coverage, down to ×0.80 penalty for under 40%
- **Explainability Engine** — 6–8 plain-English lines explaining every score
- **5 Scoring Profiles** — Balanced, Senior/Lead, Fresh Graduate, Technical, Management
- **Keyword Stuffing Penalty** — penalises resumes that game keyword density
- **Resume Length Normalisation** — penalises sparse/incomplete resumes
- **Section-Aware Skill Bonus** — rewards structured resumes with a dedicated skills section
- **CSV Export** — download full ranking report
- **Radar Charts** — per-candidate score visualisation

---

## Project Structure

```
resume_ranker/
├── app.py                  # Flask web application & API routes
├── scoring.py              # Core ranking engine (TF-IDF, 5-signal scorer)
├── nlp_utils.py            # NLP utilities (preprocessing, skill/exp/edu extraction)
├── config.py               # Skill taxonomy, weight profiles, grade thresholds
├── jd_analyzer.py          # JD Quality Scorer + Bias Detector
├── red_flag_detector.py    # Resume Red Flag & Green Flag Detector
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Full frontend (HTML + CSS + JS, single file)
├── uploads/                # Temp storage for uploaded resumes (auto-deleted)
└── reports/                # JSON ranking reports (for CSV download)
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Step 1 — Clone or extract the project
```bash
cd "C:\Users\YourName\Downloads\resume_ranker"
```

### Step 2 — Install dependencies
```bash
pip install flask pdfplumber scikit-learn pandas werkzeug python-docx
```

### Step 3 — (Optional but recommended) Install spaCy for better NLP
```bash
pip install spacy
python -m spacy download en_core_web_sm
```
> Without spaCy the system uses a rule-based fallback. With spaCy scores improve by ~10–15%.

### Step 4 — Run the app
```bash
python app.py
```

### Step 5 — Open in browser
```
http://127.0.0.1:5000
```

---

## How to Use

### Ranking Resumes
1. Paste a job description into the **Job Description** field
2. Upload resume files (PDF, TXT, or DOCX) via drag & drop or click to browse
3. Select a **Scoring Profile** (Balanced, Senior, Fresher, Technical, Management)
4. Click **⚡ Rank All Candidates**
5. Expand any candidate card to see skill breakdown, radar chart, and explanation

### JD Quality Scorer
1. Paste a job description
2. Click **🔍 Score JD Quality** (before ranking)
3. Get an A–F grade with improvement suggestions and bias flags

### Red Flag Detector
1. Rank your resumes first
2. Expand any candidate card
3. Click **🚩 Scan Red Flags** at the bottom of the card

### Candidate Comparison
1. Rank at least 2 resumes
2. Click **⚖️ Compare Candidates** in the results bar
3. Select two candidates from the dropdowns
4. Click **Compare →**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main UI |
| POST | `/rank` | Rank all uploaded resumes against JD |
| POST | `/analyze_jd` | Score a job description for quality & bias |
| POST | `/red_flags` | Scan a resume text for red flags |
| POST | `/compare` | Compare two candidates head-to-head |
| GET | `/download/<id>` | Download ranking report as CSV |
| GET | `/status` | Check NLP library availability |

---

## Scoring Profiles

| Profile | TF-IDF | Keywords | Skills | Experience | Education |
|---------|--------|----------|--------|------------|-----------|
| Balanced | 30% | 25% | 25% | 12% | 8% |
| Senior / Lead | 25% | 20% | 20% | 25% | 10% |
| Fresh Graduate | 30% | 25% | 25% | 5% | 15% |
| Technical | 20% | 20% | 40% | 12% | 8% |
| Management | 30% | 30% | 15% | 15% | 10% |

---

## Grade Scale

| Score | Grade | Label |
|-------|-------|-------|
| 85–100 | A+ | Outstanding Match |
| 75–84 | A | Excellent Match |
| 65–74 | B | Good Match |
| 50–64 | C | Moderate Match |
| 35–49 | D | Weak Match |
| 0–34 | F | Poor Match |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12, Flask 3.0 |
| NLP | spaCy 3.7.4, scikit-learn (TF-IDF) |
| ML | Cosine Similarity, TF-IDF Vectoriser |
| File Parsing | pdfplumber (PDF), python-docx (DOCX) |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | HTML5 Canvas (custom radar chart) |
| Data | pandas, NumPy |
| Security | Werkzeug secure_filename |

---

## Troubleshooting

**App won't start — signal error on Windows**
```python
# In app.py, change:
app.run(debug=True, port=5000, use_reloader=False)  # add use_reloader=False
```

**TemplateNotFound: index.html**
Make sure `index.html` is inside a `templates/` subfolder, not in the root folder.

**PDF text not extracting**
Install pdfplumber: `pip install pdfplumber`
Scanned PDFs (image-only) cannot be extracted — use text-based PDFs.

**Low scores across all resumes**
This is normal when the JD and resumes have very different vocabulary. Try the spaCy installation for better lemmatisation.

---

## Test Data

Sample test files are included for quick demonstration:

| File | Description |
|------|-------------|
| `jd_hiring_manager.txt` | Senior Hiring Manager JD — NovaTech Solutions |
| `resume_meera_krishnamurthy.txt` | #1 expected — 9yr IIM MBA, Flipkart, SHRM-SCP |
| `resume_arjun_venkatesh.txt` | #2 expected — 7yr XLRI MBA, Swiggy, SHRM-CP |
| `resume_divya_subramaniam.txt` | #3 expected — 6yr, smaller team scope |
| `resume_rahul_mehrotra.txt` | #4 expected — 8yr but FMCG/non-tech HR |
| `resume_sneha_pillai.txt` | #5 expected — 3yr, no leadership |
| `resume_karthik_balaji.txt` | #6 expected — Software engineer, wrong domain |

---

## Author

Built as an internship project — 2026.
