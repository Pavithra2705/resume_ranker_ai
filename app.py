"""
app.py — ResumeRank AI v2
Flask web application — thin layer over scoring engine.
"""

import os, json, io, re, traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from scoring import rank_all_resumes
from config import WEIGHT_PROFILES, DEFAULT_PROFILE, ALLOWED_EXTENSIONS, AUTO_DELETE_UPLOADS
from jd_analyzer import analyze_jd
from red_flag_detector import detect_red_flags

# ── Optional deps ────────────────────────────────────────────────────────────
try:
    import pdfplumber;  PDF_OK = True
except ImportError:
    PDF_OK = False
    print("[WARN] pdfplumber missing → pip install pdfplumber")

try:
    import pandas as pd; PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    from docx import Document; DOCX_OK = True
except ImportError:
    DOCX_OK = False

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs('uploads', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        if not PDF_OK:
            return ""
        text = ""
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        except Exception as e:
            print(f"[PDF ERROR] {e}")
        return text.strip()

    elif ext == '.docx':
        if not DOCX_OK:
            return ""
        try:
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print(f"[DOCX ERROR] {e}")
            return ""

    elif ext in ['.txt', '.md']:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    return ""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', profiles=WEIGHT_PROFILES)

@app.route('/rank', methods=['POST'])
def rank():
    try:
        jd = request.form.get('job_description', '').strip()
        profile = request.form.get('profile', DEFAULT_PROFILE)

        if not jd:
            return jsonify({'error': 'Job description is required'}), 400

        files = request.files.getlist('resumes')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'Upload at least one resume'}), 400

        resumes = []
        errors = []
        saved_paths = []

        for f in files:
            if not f.filename:
                continue
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                errors.append(f"{f.filename}: unsupported format ({ext})")
                continue

            safe = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
            f.save(path)
            saved_paths.append(path)

            text = extract_text(path)
            if not text or len(text.strip()) < 30:
                errors.append(f"{f.filename}: could not extract text (is it a scanned PDF?)")
                continue

            resumes.append({"filename": f.filename, "text": text})

        if not resumes:
            return jsonify({'error': 'No usable resumes. ' + ' | '.join(errors)}), 400

        # ── Run ranking engine ─────────────────────────────────────────────
        results = rank_all_resumes(jd, resumes, profile=profile)

        # ── Clean up uploaded files ────────────────────────────────────────
        if AUTO_DELETE_UPLOADS:
            for p in saved_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

        # ── Persist report ─────────────────────────────────────────────────
        report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join('reports', f'report_{report_id}.json')

        # Remove non-serializable keys before saving
        safe_results = []
        for r in results:
            safe_results.append({k: v for k, v in r.items() if isinstance(v, (str, int, float, list, bool))})

        with open(report_path, 'w') as rf:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "job_description": jd[:500],
                "profile": profile,
                "results": safe_results,
                "errors": errors
            }, rf, indent=2)

        return jsonify({
            'results': safe_results,
            'report_id': report_id,
            'total': len(results),
            'profile': WEIGHT_PROFILES.get(profile, {}).get("label", profile),
            'warnings': errors
        })

    except Exception as e:
        print(f"[FATAL] /rank: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/download/<report_id>')
def download(report_id):
    path = os.path.join('reports', f'report_{report_id}.json')
    if not os.path.exists(path):
        return "Report not found", 404

    with open(path) as f:
        data = json.load(f)

    rows = []
    for r in data['results']:
        rows.append({
            "Rank":             r.get("rank"),
            "Filename":         r.get("filename"),
            "Final Score":      r.get("final_score"),
            "Grade":            r.get("grade"),
            "Label":            r.get("label"),
            "TF-IDF (%)":       r.get("tfidf_pct"),
            "Keyword Match (%)":r.get("keyword_pct"),
            "Skill Match (%)":  r.get("skill_pct"),
            "Experience (yrs)": r.get("years_experience"),
            "Education":        r.get("education_label"),
            "Skills Found":     ", ".join(r.get("skills_found", [])),
            "Skills Matched":   ", ".join(r.get("skills_matched", [])),
            "Skills Missing":   ", ".join(r.get("skills_missing", [])),
        })

    if PANDAS_OK:
        df = pd.DataFrame(rows)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv = buf.getvalue()
    else:
        headers = list(rows[0].keys())
        lines = [",".join(f'"{h}"' for h in headers)]
        for row in rows:
            lines.append(",".join(f'"{str(v)}"' for v in row.values()))
        csv = "\n".join(lines)

    return send_file(
        io.BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'resume_ranking_{report_id}.csv'
    )


@app.route('/status')
def status():
    from nlp_utils import SPACY_OK
    try:
        from sklearn import __version__ as skv; SK_OK = True
    except Exception:
        SK_OK = False
    return jsonify({
        "spacy": SPACY_OK, "sklearn": SK_OK,
        "pdfplumber": PDF_OK, "pandas": PANDAS_OK,
        "python-docx": DOCX_OK,
    })


# ── NEW: JD Quality Scorer ────────────────────────────────────────────────────
@app.route('/analyze_jd', methods=['POST'])
def analyze_jd_route():
    try:
        data = request.get_json()
        jd = (data or {}).get('job_description', '').strip()
        if not jd:
            return jsonify({'error': 'No job description provided'}), 400
        result = analyze_jd(jd)
        return jsonify(result)
    except Exception as e:
        print(f"[FATAL] /analyze_jd: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# ── NEW: Red Flag Detector ────────────────────────────────────────────────────
@app.route('/red_flags', methods=['POST'])
def red_flags_route():
    try:
        data = request.get_json()
        resume_text = (data or {}).get('resume_text', '').strip()
        filename    = (data or {}).get('filename', 'resume')
        if not resume_text:
            return jsonify({'error': 'No resume text provided'}), 400
        result = detect_red_flags(resume_text)
        result['filename'] = filename
        return jsonify(result)
    except Exception as e:
        print(f"[FATAL] /red_flags: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# ── NEW: Candidate Comparison ─────────────────────────────────────────────────
@app.route('/compare', methods=['POST'])
def compare_route():
    try:
        data = request.get_json()
        results  = (data or {}).get('results', [])
        idx_a    = (data or {}).get('idx_a', 0)
        idx_b    = (data or {}).get('idx_b', 1)

        if len(results) < 2:
            return jsonify({'error': 'Need at least 2 candidates to compare'}), 400

        a = results[idx_a]
        b = results[idx_b]

        SIGNALS = ['tfidf_pct', 'keyword_pct', 'skill_pct', 'experience_pct', 'education_pct']
        LABELS  = ['TF-IDF Similarity', 'Keyword Match', 'Skill Overlap', 'Experience', 'Education']

        breakdown = []
        for sig, lbl in zip(SIGNALS, LABELS):
            va = a.get(sig, 0)
            vb = b.get(sig, 0)
            diff = round(va - vb, 1)
            winner = 'a' if va > vb else 'b' if vb > va else 'tie'
            breakdown.append({
                "signal":  lbl,
                "a_val":   va,
                "b_val":   vb,
                "diff":    abs(diff),
                "winner":  winner,
            })

        # Skills unique to each
        skills_a = set(a.get('skills_found', []))
        skills_b = set(b.get('skills_found', []))
        unique_a = sorted(skills_a - skills_b)
        unique_b = sorted(skills_b - skills_a)
        common   = sorted(skills_a & skills_b)

        # Overall recommendation
        score_a = a.get('final_score', 0)
        score_b = b.get('final_score', 0)
        margin  = abs(score_a - score_b)

        if margin < 3:
            recommendation = f"Near tie ({score_a} vs {score_b}) — interview both and assess culture fit"
        elif score_a > score_b:
            recommendation = f"{a['filename']} is the stronger candidate (+{round(margin,1)} pts) — prioritize for interview"
        else:
            recommendation = f"{b['filename']} is the stronger candidate (+{round(margin,1)} pts) — prioritize for interview"

        return jsonify({
            "candidate_a":     a,
            "candidate_b":     b,
            "breakdown":       breakdown,
            "unique_to_a":     unique_a[:10],
            "unique_to_b":     unique_b[:10],
            "common_skills":   common[:10],
            "recommendation":  recommendation,
            "margin":          round(margin, 1),
        })
    except Exception as e:
        print(f"[FATAL] /compare: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    from nlp_utils import SPACY_OK
    print("\n" + "="*55)
    print("  ResumeRank AI v2 — Starting")
    print("="*55)
    print(f"  SpaCy NLP:    {'✓ Active' if SPACY_OK  else '⚠ Fallback (no spacy)'}")
    print(f"  scikit-learn: {'✓ Active' if True       else '✗ Missing'}")
    print(f"  pdfplumber:   {'✓ Active' if PDF_OK     else '⚠ PDFs disabled'}")
    print(f"  python-docx:  {'✓ Active' if DOCX_OK    else '⚠ DOCX disabled'}")
    print("="*55)
    print("  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
