# AI Resume Screening System

A Streamlit app that ranks candidate resumes against a job description using TF-IDF similarity and keyword-based skill matching.

## What's new in the UI
- Styled header, score cards, and color-coded match badges (green/yellow/red)
- Summary metrics row (candidates screened, top score, average score, strong matches)
- Progress bar while resumes are being analyzed
- Cleaner detailed-analysis cards with per-candidate metrics and a score progress bar
- One-click CSV download of the ranking table

## Files
- `main.py` — the app
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run main.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a GitHub repo (e.g. `resume-screener`) and push these two files (`main.py`, `requirements.txt`) to it.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, and set the main file path to `main.py`.
4. Click **Deploy**. Streamlit installs dependencies from `requirements.txt` automatically.
5. You'll get a public link like `https://your-app-name.streamlit.app` — that's the one to share.

Redeploys happen automatically whenever you push new commits to the repo.
