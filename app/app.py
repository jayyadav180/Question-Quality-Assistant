import streamlit as st
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feedback import generate_feedback
from src.model    import load_model, predict_score

# ─────────────────────────────────────────────────────
# PAGE CONFIG — must be first streamlit command
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Question Quality Assistant",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "collapsed"
)

# ─────────────────────────────────────────────────────
# LOAD MODEL ONCE
# ─────────────────────────────────────────────────────

@st.cache_resource
def get_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'models', 'classifier.pkl'
    )
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

model = get_model()

# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────

st.title("🔍 Question Quality Assistant")
st.markdown(
    "Analyze your Stack Overflow question **before you post it**. "
    "Get a quality score and specific suggestions instantly."
)

if model:
    st.success("✅ ML model loaded — ready to analyze")
else:
    st.warning(
        "⚠️ Model not found. "
        "Run `python src/model.py` to train it first."
    )

st.divider()

# ─────────────────────────────────────────────────────
# INPUT SECTION
# Two columns: inputs left, tips right
# ─────────────────────────────────────────────────────

col_input, col_tips = st.columns([2, 1])

with col_input:
    st.subheader("📝 Your Draft Question")

    title = st.text_input(
        label       = "Question Title",
        placeholder = (
            "e.g. TypeError when calling .split() "
            "on pandas column after merge"
        ),
        help = (
            "6–15 words. Include technology name, "
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
            "4. Your code (wrap in triple backticks ```)\n"
            "5. What you expected vs what happened"
        ),
        height = 300,
        help   = "50–600 words works best."
    )

    # ── Formatting helper expander ──
    # This sits directly below the body text area
    # at the same indentation level — NOT inside body
    with st.expander("📋 How to format code in your question"):
        st.markdown(
            "Wrap your code in triple backticks "
            "for accurate scoring:\n\n"
            "\\`\\`\\`python\n"
            "your code here\n"
            "\\`\\`\\`\n\n"
            "**Why it matters:** Properly formatted code "
            "is easier to read and signals question quality "
            "to both reviewers and this system."
        )

    tags = st.text_input(
        label       = "Tags (optional)",
        placeholder = "e.g. <python><pandas><dataframe>",
        help        = "Add 2–5 relevant technology tags."
    )

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
# Only runs when button is clicked
# ─────────────────────────────────────────────────────

if analyze:

    # ── Input validation ──
    if not title.strip():
        st.error("Please enter a question title.")
        st.stop()
    if not body.strip():
        st.error("Please enter a question body.")
        st.stop()

    # ── Unformatted code detection ──
    # Checks if body contains code signals but no
    # formatting markers — warns user before results
    has_import   = bool(re.search(r'\bimport\s+\w+', body))
    has_def      = bool(re.search(
                        r'\bdef\s+\w+\s*\(', body))
    has_backtick = '```' in body or '<code>' in body

    unformatted_code_likely = (
        (has_import or has_def) and not has_backtick
    )

    if unformatted_code_likely:
        st.info(
            "💡 **Tip:** Your body appears to contain "
            "code that is not wrapped in backticks. "
            "Wrap it in \\`\\`\\` for accurate scoring "
            "and better Stack Overflow formatting."
        )

    # ── Run both systems ──
    with st.spinner("Analyzing your question..."):
        rule_result = generate_feedback(title, body, tags)

        ml_result = None
        if model:
            ml_result = predict_score(
                model, title, body, tags
            )

    st.divider()
    st.subheader("📊 Quality Assessment")

    # ── Score metrics ──
    if ml_result:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "🎯 Rule-Based Score",
                f"{rule_result['overall_score']}%",
                help="Structural feature analysis"
            )
        with c2:
            st.metric(
                "🤖 ML Answerability",
                f"{ml_result['overall_score']}%",
                help=(
                    "Learned from 30k "
                    "Stack Overflow questions"
                )
            )
        with c3:
            st.metric(
                "📌 Title Score",
                f"{rule_result['title_score']}%"
            )
        with c4:
            st.metric(
                "📄 Body Score",
                f"{rule_result['body_score']}%"
            )
    else:
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

    # ── Progress bars ──
    st.progress(
        rule_result['title_score'] / 100,
        text = (
            f"Title Quality: "
            f"{rule_result['title_score']}%"
        )
    )
    st.progress(
        rule_result['body_score'] / 100,
        text = (
            f"Body Quality: "
            f"{rule_result['body_score']}%"
        )
    )

    # ── Quality tier badge ──
    tier_icons = {
        "Excellent":  "🟢",
        "Good":       "🟡",
        "Needs Work": "🟠",
        "Poor":       "🔴"
    }
    icon = tier_icons.get(rule_result['tier'], "⚪")

    if rule_result['tier_color'] == 'success':
        st.success(
            f"{icon} **{rule_result['tier']}** — "
            f"{rule_result['tier_message']}"
        )
    elif rule_result['tier_color'] == 'warning':
        st.warning(
            f"{icon} **{rule_result['tier']}** — "
            f"{rule_result['tier_message']}"
        )
    else:
        st.error(
            f"{icon} **{rule_result['tier']}** — "
            f"{rule_result['tier_message']}"
        )

    st.divider()

    # ── Warnings — high priority ──
    if rule_result['warnings']:
        st.subheader("⚠️ Issues To Fix Before Posting")
        for w in rule_result['warnings']:
            st.error(
                f"**{w['dimension']}:** {w['message']}"
            )

    # ── Suggestions — medium/low priority ──
    if rule_result['feedback']:
        st.subheader("💡 Suggestions To Improve Your Score")
        for fb in rule_result['feedback']:
            st.warning(
                f"**{fb['dimension']}:** {fb['message']}"
            )

    # ── Everything good ──
    if (not rule_result['warnings'] and
            not rule_result['feedback']):
        st.success(
            "✅ Your question is well-structured. "
            "You are ready to post!"
        )

    # ── ML confidence breakdown ──
    if ml_result:
        with st.expander(
            "🤖 ML Model Confidence Breakdown"
        ):
            st.markdown(
                "Trained on 30,000 Stack Overflow "
                "questions (15k HQ, 15k LQ_CLOSE). "
                "Predicts answerability from "
                "structural features."
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

    # ── Feature analysis ──
    with st.expander("🔬 Detailed Feature Analysis"):

        # Show detected archetype
        archetype_labels = {
            'debugging':  '🔴 Debugging — error/crash',
            'behavioral': '🟡 Behavioral — unexpected behavior',
            'conceptual': '🔵 Conceptual — understanding',
            'howto':      '🟢 How-To — approach question',
            'unknown':    '⚪ Unknown — mixed type'
        }
        detected = rule_result['features'].get(
            'detected_archetype', 'unknown'
        )
        st.info(
            f"**Detected Question Type:** "
            f"{archetype_labels.get(detected, detected)}"
        )

        features = rule_result['features']
        f_col1, f_col2 = st.columns(2)

        title_features = {
            'title_word_count':
                features['title_word_count'],
            'title_has_technology':
                features['title_has_technology'],
            'title_has_error_type':
                features['title_has_error_type'],
            'title_is_vague':
                features['title_is_vague'],
        }

        body_features = {
            'has_code_block':
                1 if (
                    '<code>' in str(body) or
                    '```'    in str(body)
                ) else 0,
            'body_word_count':
                features['body_word_count'],
            'body_length_adequate':
                features['body_length_adequate'],
            'code_to_text_ratio':
                features['code_to_text_ratio'],
            'has_error_keywords':
                features['has_error_keywords'],
            'has_colloquial_error':
                features['has_colloquial_error'],
            'has_attempt_signal':
                features['has_attempt_signal'],
            'has_expected_vs_actual':
                features['has_expected_vs_actual'],
            'has_question_mark':
                features['has_question_mark'],
            'tag_count':
                features['tag_count'],
        }

        with f_col1:
            st.markdown("**Title Features**")
            for k, v in title_features.items():
                if k == 'title_word_count':
                    st.write(f"📏 {k}: **{v} words**")
                elif k == 'title_is_vague':
                    icon = "⚠️" if v else "✅"
                    st.write(f"{icon} {k}: **{v}**")
                else:
                    icon = "✅" if v else "❌"
                    st.write(f"{icon} {k}: **{v}**")

        with f_col2:
            st.markdown("**Body Features**")
            for k, v in body_features.items():
                if k in [
                    'body_word_count',
                    'code_to_text_ratio',
                    'tag_count'
                ]:
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
    "Accuracy: 66% on balanced binary classification."
)