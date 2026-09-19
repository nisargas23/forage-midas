import streamlit as st
import pandas as pd
import numpy as np
import re
import os

from pypdf import PdfReader
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .score-card {
        background-color: #f8f9fb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-strong { background-color: #dcfce7; color: #166534; }
    .badge-moderate { background-color: #fef9c3; color: #854d0e; }
    .badge-low { background-color: #fee2e2; color: #991b1b; }
    .section-divider {
        margin: 1.5rem 0 1rem 0;
        border: none;
        border-top: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def recommendation_badge(recommendation):
    css_class = {
        "Strong Match": "badge-strong",
        "Moderate Match": "badge-moderate",
        "Low Match": "badge-low",
    }.get(recommendation, "badge-moderate")

    return f'<span class="badge {css_class}">{recommendation}</span>'


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown('<div class="main-header">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload resumes and enter a job description to automatically '
    'rank candidates based on skill and job-description similarity.</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# RESUME TEXT EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(file):
    """
    Extract text from PDF file
    """
    text = ""

    try:
        reader = PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:
        st.error(f"Error reading PDF: {e}")

    return text


def extract_text_from_docx(file):
    """
    Extract text from DOCX file
    """
    text = ""

    try:
        document = Document(file)

        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"

    except Exception as e:
        st.error(f"Error reading DOCX: {e}")

    return text


def extract_text_from_txt(file):
    """
    Extract text from TXT file
    """
    try:
        return file.read().decode("utf-8")

    except Exception as e:
        st.error(f"Error reading TXT file: {e}")

        return ""


# --------------------------------------------------
# GENERIC TEXT EXTRACTION
# --------------------------------------------------

def extract_resume_text(file):

    file_name = file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file)

    elif file_name.endswith(".docx"):
        return extract_text_from_docx(file)

    elif file_name.endswith(".txt"):
        return extract_text_from_txt(file)

    else:
        return ""


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

def clean_text(text):

    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9+#.]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# SKILL DATABASE
# --------------------------------------------------

SKILLS = [

    # Programming
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "javascript",

    # Software
    "data structures",
    "algorithms",
    "oops",
    "object oriented programming",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "ai",
    "tensorflow",
    "pytorch",
    "scikit learn",

    # Web
    "html",
    "css",
    "react",
    "node.js",

    # Database
    "mysql",
    "mongodb",

    # Embedded / EEE
    "embedded systems",
    "iot",
    "arduino",
    "esp32",
    "microcontroller",
    "matlab",
    "simulink",
    "pcb",
    "verilog",

    # Engineering
    "electrical engineering",
    "electronics",
    "power electronics",
    "control systems",
    "circuit design"
]


# --------------------------------------------------
# SKILL EXTRACTION
# --------------------------------------------------

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        if skill.lower() in text:

            found_skills.append(skill)

    return list(set(found_skills))


# --------------------------------------------------
# EXPERIENCE EXTRACTION
# --------------------------------------------------

def extract_experience(text):

    text = text.lower()

    patterns = [

        r"(\d+)\+?\s*years?\s*of\s*experience",

        r"(\d+)\+?\s*years?\s*experience",

        r"experience\s*:\s*(\d+)\s*years?"

    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:

            return int(match.group(1))

    return 0


# --------------------------------------------------
# AI MATCH SCORE
# --------------------------------------------------

def calculate_similarity(job_description, resume_text):

    documents = [

        clean_text(job_description),

        clean_text(resume_text)

    ]

    vectorizer = TfidfVectorizer(

        stop_words="english",

        ngram_range=(1, 2)

    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(

        tfidf_matrix[0:1],

        tfidf_matrix[1:2]

    )[0][0]

    return round(similarity * 100, 2)


# --------------------------------------------------
# SKILL MATCH SCORE
# --------------------------------------------------

def calculate_skill_match(job_description, resume_text):

    required_skills = extract_skills(job_description)

    resume_skills = extract_skills(resume_text)

    if len(required_skills) == 0:

        return 0, [], resume_skills

    matched_skills = [

        skill

        for skill in required_skills

        if skill in resume_skills

    ]

    skill_score = (

        len(matched_skills)

        / len(required_skills)

    ) * 100

    return (

        round(skill_score, 2),

        matched_skills,

        resume_skills

    )


# --------------------------------------------------
# FINAL SCORE
# --------------------------------------------------

def calculate_final_score(

    similarity_score,

    skill_score

):

    # Weightage

    similarity_weight = 0.6

    skill_weight = 0.4

    final_score = (

        similarity_score

        * similarity_weight

        +

        skill_score

        * skill_weight

    )

    return round(final_score, 2)


# --------------------------------------------------
# SIDEBAR: JOB DETAILS
# --------------------------------------------------

st.sidebar.markdown("## 🧾 Job Details")

job_title = st.sidebar.text_input(
    "Job Title",
    "Software Developer"
)

job_description = st.sidebar.text_area(
    "Job Description",
    """
    We are looking for a Software Developer.

    Required skills:

    Python
    Java
    SQL
    Data Structures
    Algorithms
    Object Oriented Programming
    Machine Learning

    Candidates should have good programming
    and problem-solving skills.
    """,
    height=280
)

with st.sidebar.expander("ℹ️ How scoring works"):
    st.write(
        "- **AI Similarity (60%)** — TF-IDF cosine similarity between the "
        "resume and job description.\n"
        "- **Skill Match (40%)** — share of required skills found in the resume.\n"
        "- **75%+** → Strong Match, **50–74%** → Moderate Match, "
        "**below 50%** → Low Match."
    )


# --------------------------------------------------
# RESUME UPLOAD
# --------------------------------------------------

st.markdown("### 📤 Upload Resumes")

uploaded_files = st.file_uploader(
    "Drag and drop resume files here, or click to browse",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    st.caption(f"{len(uploaded_files)} file(s) ready — {', '.join(f.name for f in uploaded_files)}")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

screen_clicked = st.button("🚀 Screen Resumes", type="primary", use_container_width=False)


# --------------------------------------------------
# SCREEN RESUMES
# --------------------------------------------------

if screen_clicked:

    if not job_description.strip():

        st.warning(

            "Please enter a job description."

        )

    elif not uploaded_files:

        st.warning(

            "Please upload at least one resume."

        )

    else:

        results = []

        progress_bar = st.progress(0, text="Starting screening...")

        # Process every resume

        for i, file in enumerate(uploaded_files):

            progress_bar.progress(
                (i) / len(uploaded_files),
                text=f"Analyzing {file.name}..."
            )

            resume_text = extract_resume_text(file)


            if not resume_text.strip():

                continue


            # AI similarity

            similarity_score = calculate_similarity(

                job_description,

                resume_text

            )


            # Skills

            skill_score, matched_skills, resume_skills = (

                calculate_skill_match(

                    job_description,

                    resume_text

                )

            )


            # Final score

            final_score = calculate_final_score(

                similarity_score,

                skill_score

            )


            # Experience

            experience = extract_experience(

                resume_text

            )


            # Recommendation

            if final_score >= 75:

                recommendation = "Strong Match"

            elif final_score >= 50:

                recommendation = "Moderate Match"

            else:

                recommendation = "Low Match"


            results.append({

                "Candidate": file.name,

                "AI Similarity": similarity_score,

                "Skill Match": skill_score,

                "Final Score": final_score,

                "Experience": experience,

                "Matched Skills": ", ".join(

                    matched_skills

                ),

                "Recommendation": recommendation

            })

        progress_bar.progress(1.0, text="Done!")
        progress_bar.empty()


        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------

        if results:

            df = pd.DataFrame(results)


            # Sort by final score

            df = df.sort_values(

                by="Final Score",

                ascending=False

            )


            # Rank

            df.insert(

                0,

                "Rank",

                range(1, len(df) + 1)

            )


            st.success(

                f"Successfully screened {len(df)} resume(s) for **{job_title}**."

            )

            # Top candidate + summary metrics

            top_candidate = df.iloc[0]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Candidates Screened", len(df))
            m2.metric("Top Score", f"{top_candidate['Final Score']}%")
            m3.metric("Avg. Score", f"{round(df['Final Score'].mean(), 1)}%")
            m4.metric(
                "Strong Matches",
                int((df["Recommendation"] == "Strong Match").sum())
            )

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            st.markdown("### 🏆 Top Candidate")

            st.markdown(
                f"""
                <div class="score-card">
                    <b>{top_candidate['Candidate']}</b><br>
                    Final Score: <b>{top_candidate['Final Score']}%</b>
                    &nbsp;&nbsp;{recommendation_badge(top_candidate['Recommendation'])}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Results table

            st.markdown("### 📊 Candidate Ranking")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_data,
                file_name="resume_screening_results.csv",
                mime="text/csv"
            )

            # Score chart

            st.markdown("### 📈 Candidate Scores")

            chart_data = df.set_index(

                "Candidate"

            )[

                [

                    "AI Similarity",

                    "Skill Match",

                    "Final Score"

                ]

            ]


            st.bar_chart(

                chart_data

            )


            # Detailed analysis

            st.markdown("### 🔍 Detailed Candidate Analysis")


            for _, row in df.iterrows():

                with st.expander(
                    f"{row['Rank']}. {row['Candidate']} — {row['Final Score']}%"
                ):

                    st.markdown(recommendation_badge(row["Recommendation"]), unsafe_allow_html=True)
                    st.write("")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("AI Similarity", f"{row['AI Similarity']}%")
                    c2.metric("Skill Match", f"{row['Skill Match']}%")
                    c3.metric("Experience", f"{row['Experience']} yrs")

                    st.progress(min(int(row["Final Score"]), 100) / 100)

                    st.write(f"**Matched Skills:** {row['Matched Skills'] or 'None found'}")


        else:

            st.error(

                "Unable to extract text from resumes."

            )

else:
    st.info("👋 Enter a job description in the sidebar, upload resumes above, then click **Screen Resumes**.")
