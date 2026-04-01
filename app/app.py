# app/app.py
#
# This is your complete Streamlit frontend.
# Streamlit converts this Python script into a
# full interactive web interface in your browser.
#
# HOW TO RUN:
#   streamlit run app/app.py
#
# This file connects all three src/ modules:
#   features.py  →  extract_features()
#   model.py     →  load_model(), predict_score()
#   feedback.py  →  generate_feedback()

import streamlit as st
import sys
import os
import joblib
import pandas as pd

# Make src/ folder importable from app/ folder
# Without this, Python cannot find your modules
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..'))

from src.feedback import generate_feedback
from src.model    import load_model, predict_score

# ─────────────────────────────────────────────────────
# PAGE CONFIGURATION
# Must be the first Streamlit command in the file
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Question Quality Assistant",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "collapsed"
)

# ─────────────────────────────────────────────────────
# LOAD MODEL ONCE AT STARTUP
# st.cache_resource means this only runs once —
# not on every user interaction. Critical for performance.
# ─────────────────────────────────────────────────────

@st.cache_resource
def get_model():
    """
    Loads the trained classifier from disk once
    and caches it for the entire session.
    Without caching, the model would reload on
    every button click — very slow.
    """
    model_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'models', 'classifier.pkl'
    )
    if os.path.exists(model_path):
        return load_model(model_path)
    return None   # graceful fallback if not trained yet


model = get_model()

# ─────────────────────────────────────────────────────
# HEADER SECTION
# ─────────────────────────────────────────────────────

st.title("🔍 Question Quality Assistant")
st.markdown(
    "Analyze your Stack Overflow question **before you "
    "post it**. Get a quality score and specific "
    "suggestions to improve it instantly."
)

# Model status indicator — transparent to the user
if model:
    st.success("✅ ML model loaded — ready to analyze")
else:
    st.warning(
        "⚠️ Model not found. Run `python src/model.py` "
        "to train it first. Rule-based feedback still works."
    )

st.divider()

# ─────────────────────────────────────────────────────
# INPUT SECTION
# This is where users write their draft question.
# Two columns: input on left, live tips on right.
# ─────────────────────────────────────────────────────

col_input, col_tips = st.columns([2, 1])

with col_input:
    st.subheader("📝 Your Draft Question")

    title = st.text_input(
        label       = "Question Title",
        placeholder = (
            "e.g. TypeError when calling .split() on "
            "pandas column after merge"
        ),
        help = (
            "6–15 words. Include: technology name, "
            "what you were doing, and what went wrong."
        )
    )

    body = st.text_area(
        label       = "Question Body",
        placeholder = (
            "Describe your problem here...\n\n"
            "Include:\n"
            "1. What you were trying to accomplish\n"
            "2. What you tried\n"
            "3. The exact error message\n"
            "4. Your code (minimal example)\n"
            "5. What you expected vs what happened"
        ),
        height = 300,
        help   = (
            "50–600 words works best. "
            "Paste your actual error message and code."
        )
    )

    tags = st.text_input(
        label       = "Tags (optional)",
        placeholder = "e.g. <python><pandas><dataframe>",
        help        = "Add 2–5 relevant technology tags."
    )

    # The analyze button — triggers entire pipeline
    analyze = st.button(
        "🔍 Analyze My Question",
        type = "primary",
        use_container_width = True
    )

with col_tips:
    st.subheader("💡 Quick Tips")
    st.info(
        "**Strong title formula:**\n\n"
        "`[Technology]` + `[Action]` + `[Error]`\n\n"
        "Example: *Python pandas merge drops rows — "
        "TypeError on .split()*"
    )
    st.info(
        "**Body checklist:**\n\n"
        "☐ Minimal code example\n\n"
        "☐ Exact error message\n\n"
        "☐ What you already tried\n\n"
        "☐ Expected vs actual output"
    )
    st.info(
        "**Score guide:**\n\n"
        "🟢 85–100% — Ready to post\n\n"
        "🟡 70–84% — Minor improvements\n\n"
        "🟠 50–69% — Needs work\n\n"
        "🔴 0–49% — Significant revision"
    )

# ─────────────────────────────────────────────────────
# ANALYSIS PIPELINE
# Runs when user clicks the button.
# Calls both ML model and rule-based feedback,
# then displays everything in a structured layout.
# ─────────────────────────────────────────────────────

if analyze:

    # Input validation — gentle, not harsh
    if not title.strip():
        st.error("Please enter a question title.")
        st.stop()
    if not body.strip():
        st.error("Please enter a question body.")
        st.stop()

    # Run both systems simultaneously
    with st.spinner("Analyzing your question..."):

        # Rule-based feedback — always runs
        rule_result = generate_feedback(title, body, tags)

        # ML prediction — runs only if model is loaded
        ml_result = None
        if model:
            ml_result = predict_score(model, title, body, tags)

    st.divider()
    st.subheader("📊 Quality Assessment")

    # ─────────────────────────────────────────────────
    # SCORE DISPLAY
    # Three columns showing overall, title, body scores
    # ML score shown separately if model is available
    # ─────────────────────────────────────────────────

    if ml_result:
        # Four columns when ML model is available
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                label = "🎯 Rule-Based Score",
                value = f"{rule_result['overall_score']}%",
                help  = "Based on structural feature analysis"
            )
        with c2:
            st.metric(
                label = "🤖 ML Answerability",
                value = f"{ml_result['overall_score']}%",
                help  = (
                    "Probability of being answered, "
                    "learned from 30k Stack Overflow questions"
                )
            )
        with c3:
            st.metric(
                label = "📌 Title Score",
                value = f"{rule_result['title_score']}%"
            )
        with c4:
            st.metric(
                label = "📄 Body Score",
                value = f"{rule_result['body_score']}%"
            )
    else:
        # Three columns without ML model
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Overall Score",
                f"{rule_result['overall_score']}%"
            )
        with c2:
            st.metric(
                "Title Score",
                f"{rule_result['title_score']}%"
            )
        with c3:
            st.metric(
                "Body Score",
                f"{rule_result['body_score']}%"
            )

    # Visual progress bars — makes scores tangible
    st.progress(
        rule_result['title_score'] / 100,
        text = f"Title Quality: {rule_result['title_score']}%"
    )
    st.progress(
        rule_result['body_score'] / 100,
        text = f"Body Quality: {rule_result['body_score']}%"
    )

    # Quality tier badge
    tier_icons = {
        "Excellent":  "🟢",
        "Good":       "🟡",
        "Needs Work": "🟠",
        "Poor":       "🔴"
    }
    tier_icon = tier_icons.get(rule_result['tier'], "⚪")

    if rule_result['tier_color'] == 'success':
        st.success(
            f"{tier_icon} **{rule_result['tier']}** — "
            f"{rule_result['tier_message']}"
        )
    elif rule_result['tier_color'] == 'warning':
        st.warning(
            f"{tier_icon} **{rule_result['tier']}** — "
            f"{rule_result['tier_message']}"
        )
    else:
        st.error(
            f"{tier_icon} **{rule_result['tier']}** — "
            f"{rule_result['tier_message']}"
        )

    st.divider()

    # ─────────────────────────────────────────────────
    # FEEDBACK DISPLAY
    # Warnings (high priority) shown as errors in red.
    # Suggestions (medium priority) shown as warnings.
    # Combined insight rules show as info blocks.
    # ─────────────────────────────────────────────────

    has_warnings  = len(rule_result['warnings'])  > 0
    has_feedback  = len(rule_result['feedback'])  > 0

    if has_warnings:
        st.subheader("⚠️ Issues To Fix Before Posting")
        for w in rule_result['warnings']:
            st.error(
                f"**{w['dimension']}:** {w['message']}"
            )

    if has_feedback:
        st.subheader("💡 Suggestions To Improve Your Score")
        for f in rule_result['feedback']:
            st.warning(
                f"**{f['dimension']}:** {f['message']}"
            )

    # Everything looks good
    if not has_warnings and not has_feedback:
        st.success(
            "✅ Your question is well-structured and "
            "contains everything responders need. "
            "You are ready to post!"
        )

    # ─────────────────────────────────────────────────
    # ML CONFIDENCE BREAKDOWN
    # Shows the model's class probabilities transparently.
    # This is the "show your work" section — builds trust.
    # ─────────────────────────────────────────────────

    if ml_result:
        with st.expander("🤖 ML Model Confidence Breakdown"):
            st.markdown(
                "The ML model was trained on 30,000 Stack "
                "Overflow questions (15k high quality, "
                "15k closed). It learned which feature "
                "combinations predict answerability."
            )
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric(
                    "Likely to be answered",
                    f"{ml_result['p_high_quality']}%"
                )
            with mc2:
                st.metric(
                    "Likely to be closed",
                    f"{ml_result['p_lq_close']}%"
                )

    # ─────────────────────────────────────────────────
    # FEATURE ANALYSIS EXPANDER
    # Raw feature values — useful for debugging and
    # for demonstrating the system in interviews.
    # ─────────────────────────────────────────────────

    with st.expander("🔬 Detailed Feature Analysis"):
        st.markdown(
            "These are the 14 signals extracted from "
            "your question. Each one maps to a specific "
            "quality criterion identified through "
            "domain analysis of Stack Overflow "
            "community standards."
        )

        features = rule_result['features']

        # Display as two clean columns
        f_col1, f_col2 = st.columns(2)

        title_features = {
            'title_word_count':     features['title_word_count'],
            'title_has_technology': features['title_has_technology'],
            'title_has_error_type': features['title_has_error_type'],
            'title_is_vague':       features['title_is_vague'],
        }

        body_features = {
            'has_code_block':
                1 if '<code>' in str(body) or
                     '```'    in str(body) else 0,
            'body_word_count':        features['body_word_count'],
            'body_length_adequate':   features['body_length_adequate'],
            'code_to_text_ratio':     features['code_to_text_ratio'],
            'has_error_keywords':     features['has_error_keywords'],
            'has_colloquial_error':   features['has_colloquial_error'],
            'has_attempt_signal':     features['has_attempt_signal'],
            'has_expected_vs_actual': features['has_expected_vs_actual'],
            'has_question_mark':      features['has_question_mark'],
            'tag_count':              features['tag_count'],
        }

        with f_col1:
            st.markdown("**Title Features**")
            for k, v in title_features.items():
                icon = "✅" if v else "❌"
                # word counts are not binary
                if k == 'title_word_count':
                    st.write(f"📏 {k}: **{v} words**")
                elif k == 'title_is_vague':
                    # vague is bad — flip icons
                    icon = "⚠️" if v else "✅"
                    st.write(f"{icon} {k}: **{v}**")
                else:
                    st.write(f"{icon} {k}: **{v}**")

        with f_col2:
            st.markdown("**Body Features**")
            for k, v in body_features.items():
                if k in ['body_word_count',
                         'code_to_text_ratio',
                         'tag_count']:
                    st.write(f"📏 {k}: **{v}**")
                else:
                    icon = "✅" if v else "❌"
                    st.write(f"{icon} {k}: **{v}**")

# ─────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────

st.divider()
st.caption(
    "Question Quality Assistant — "
    "Built with Python, Scikit-learn, and Streamlit. "
    "Trained on 30,000 Stack Overflow questions. "
    "Model accuracy: 66% on balanced binary classification."
)