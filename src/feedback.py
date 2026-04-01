# src/feedback.py
#
# This is the rule-based judgment layer of your system.
# It runs AFTER feature extraction but INDEPENDENTLY
# of the ML model. The model gives a probability score.
# This layer gives human-readable explanations.
#
# Architecture reminder:
# features.py  →  extracts signals (numbers)
# model.py     →  predicts probability (ML)
# feedback.py  →  explains what's missing (rules)
# app.py       →  connects everything (interface)

from src.features import (
    extract_features,
    has_code_block,       # kept here for feedback rules
    clean_html,           # for text analysis
    body_word_count,
    detect_question_archetype
)


def generate_feedback(title, body, tags):
    """
    Master function that analyzes a question and returns
    a complete structured assessment including dimension
    scores, quality tier, warnings, and suggestions.
    
    This is your multidimensional scoring system —
    the core insight that makes this project a product
    rather than just a classifier.
    """

    # Extract all features first — this is our evidence
    # report that both scoring and feedback use
    f = extract_features(title, body, tags)

    archetype = detect_question_archetype(title, body)
    f['detected_archetype'] = archetype

    warnings  = []   # high priority — likely to cause closure
    feedback  = []   # medium priority — improvement suggestions

    # ─────────────────────────────────────────────────────
    # TITLE DIMENSION SCORING
    # Title contributes 35% of overall score.
    # Its job is to get the RIGHT people to click
    # and immediately understand the problem.
    # ─────────────────────────────────────────────────────

    title_score = 0

    # Length check — vague titles are almost always short
    if f['title_word_count'] < 5:
        warnings.append({
            'severity':  'high',
            'dimension': 'Title',
            'message':   (
                'Your title is too short. A strong title '
                'is 6–15 words and tells readers exactly '
                'what technology you are using and what '
                'went wrong. Example: "Pandas merge drops '
                'rows — TypeError on .split() call"'
            )
        })
    elif f['title_word_count'] > 20:
        feedback.append({
            'severity':  'medium',
            'dimension': 'Title',
            'message':   (
                'Your title is quite long. Trim it to the '
                'core problem: technology + action + symptom.'
            )
        })
    else:
        title_score += 30   # adequate length

    # Technology name — routes question to right experts
    if f['title_has_technology']:
        title_score += 35
    else:
        warnings.append({
            'severity':  'high',
            'dimension': 'Title',
            'message':   (
                'Mention the specific technology in your '
                'title — Python, React, SQL, etc. This '
                'determines which experts see your question. '
                'Without it, the right people may never '
                'find it.'
            )
        })

    # Error type name — makes title immediately specific
    if f['title_has_error_type']:
        title_score += 35
    else:
        feedback.append({
            'severity':  'medium',
            'dimension': 'Title',
            'message':   (
                'Consider naming the specific error type '
                'in your title — TypeError, 404, '
                'NullPointerException, etc. This makes '
                'your question searchable and signals '
                'you have diagnosed the problem.'
            )
        })

    # Vagueness penalty — these words add no information
    if f['title_is_vague']:
        title_score = max(0, title_score - 20)
        warnings.append({
            'severity':  'medium',
            'dimension': 'Title',
            'message':   (
                'Your title contains vague words like '
                '"help", "issue", or "not working". '
                'Replace these with the specific symptom '
                'you observed. Vague titles get fewer '
                'views and more downvotes.'
            )
        })

    # Cap at 100
    title_score = min(title_score, 100)

    # ─────────────────────────────────────────────────────
    # BODY DIMENSION SCORING
    # Body contributes 65% of overall score.
    # It carries the technical evidence a responder needs.
    # ─────────────────────────────────────────────────────

    body_score = 0

    # Code block — single strongest structural signal
    # We call has_code_block() directly here because
    # we removed it from the ML feature vector but
    # still need it for feedback logic
    if has_code_block(body):
        body_score += 25
    else:
        warnings.append({
            'severity':  'high',
            'dimension': 'Body',
            'message':   (
                'Add a minimal reproducible code example. '
                'Without code, it is nearly impossible for '
                'others to diagnose your problem. Paste '
                'the smallest snippet that demonstrates '
                'the issue.'
            )
        })

    # ── Error message rule — archetype-aware ──
    
    # ─────────────────────────────────────
    # ERROR / OUTPUT HANDLING
    # ─────────────────────────────────────

    if archetype == 'debugging':
        if f['has_error_keywords']:
            body_score += 20
        elif f['has_colloquial_error']:
            body_score += 10
            feedback.append({
                'severity':  'medium',
                'dimension': 'Body',
                'message':   (
                    'You described an error informally. '
                    'Paste the exact error message and full '
                    'stack trace for precise debugging.'
                )
            })
        else:
            warnings.append({
                'severity':  'high',
                'dimension': 'Body',
                'message':   (
                    'Include the exact error message and full '
                    'stack trace. This is essential for debugging.'
                )
            })

    elif archetype in ('behavioral', 'conceptual'):
        if not f['has_expected_vs_actual']:
            warnings.append({
                'severity':  'high',
                'dimension': 'Body',
                'message':   (
                    'Clearly describe what you expected to happen '
                    'and what actually happened.'
                )
            })

        # Optional improvement
        if not f['has_error_keywords'] and not f['has_colloquial_error']:
            feedback.append({
                'severity':  'low',
                'dimension': 'Body',
                'message':   (
                    'If your code produces console output or warnings, '
                    'include them — they can help identify the issue.'
                )
            })


    elif archetype == 'howto':
        if f['body_word_count'] < 40:
            warnings.append({
                'severity':  'medium',
                'dimension': 'Body',
                'message':   (
                    'Explain your goal more clearly. What are you trying '
                    'to achieve and why?'
                )
            })

    else:
        # Unknown — balanced fallback
        feedback.append({
            'severity':  'medium',
            'dimension': 'Body',
            'message':   (
                'Provide either an error message (if debugging) or '
                'describe the observed behavior clearly.'
            )
        })

    # Attempt signal — shows effort, prevents closure
    if f['has_attempt_signal']:
        body_score += 15
    else:
        warnings.append({
            'severity':  'medium',
            'dimension': 'Body',
            'message':   (
                'Describe what you already tried. Stack '
                'Overflow frequently closes questions '
                'that show no debugging effort. Even '
                '"I tried X and it gave Y" is enough '
                'to signal genuine effort.'
            )
        })

    # Expected vs actual — transforms complaint to question
    if f['has_expected_vs_actual']:
        body_score += 20
    else:
        feedback.append({
            'severity':  'medium',
            'dimension': 'Body',
            'message':   (
                'State what you expected to happen versus '
                'what actually happened. This simple '
                'addition makes your problem precise and '
                'immediately answerable. Example: '
                '"I expected the list to be sorted '
                'ascending but got [3, 1, 2]."'
            )
        })

    # Actual question present
    if f['has_question_mark']:
        body_score += 10
    else:
        warnings.append({
            'severity':  'low',
            'dimension': 'Body',
            'message':   (
                'Make sure you are asking an actual '
                'question. Posts without a clear question '
                'are frequently closed. End with a '
                'specific question mark sentence.'
            )
        })

    # Body length adequacy
    if f['body_length_adequate']:
        body_score += 10
    elif f['body_word_count'] < 50:
        warnings.append({
            'severity':  'high',
            'dimension': 'Body',
            'message':   (
                'Your question body is too short to give '
                'responders enough context. Explain what '
                'you were trying to accomplish, what you '
                'tried, and what went wrong.'
            )
        })
    # Note: very long bodies get no penalty here
    # because verbosity is less harmful than brevity

    body_score = min(body_score, 100)

    # ─────────────────────────────────────────────────────
    # COMBINED INSIGHT RULES
    # These fire on specific combinations of dimensions —
    # this is your multidimensional scoring logic working.
    # A single feature score cannot catch these patterns.
    # ─────────────────────────────────────────────────────

    # Good body, weak title — mismatch signal
    if body_score >= 60 and title_score < 40:
        feedback.append({
            'severity':  'medium',
            'dimension': 'Title',
            'message':   (
                'Your question body is detailed and '
                'well-structured, but your title does '
                'not reflect that quality. Rewrite your '
                'title to match the specificity of your '
                'body — a strong body deserves a '
                'strong title.'
            )
        })

    # Good title, weak body — raises false expectations
    if title_score >= 60 and body_score < 40:
        feedback.append({
            'severity':  'high',
            'dimension': 'Body',
            'message':   (
                'Your title is specific and well-formed, '
                'but your body does not deliver the '
                'context people need. A specific title '
                'raises expectations — make sure your '
                'body provides code, error messages, '
                'and what you already tried.'
            )
        })

    # Tags check — metadata quality
    if f['tag_count'] == 0:
        warnings.append({
            'severity':  'high',
            'dimension': 'Tags',
            'message':   (
                'You have no tags. Tags are how Stack '
                'Overflow routes your question to the '
                'right experts. Add 2–5 relevant tags '
                'covering the technology, framework, '
                'and topic.'
            )
        })
    elif f['tag_count'] < 2:
        feedback.append({
            'severity':  'low',
            'dimension': 'Tags',
            'message':   (
                'Consider adding more tags. Stack Overflow '
                'allows up to 5 — more relevant tags '
                'means more experts see your question.'
            )
        })

    # ─────────────────────────────────────────────────────
    # FINAL SCORE CALCULATION
    # Title 35% + Body 65% = Overall
    # Body weighted higher because it carries the
    # technical substance that determines answerability
    # ─────────────────────────────────────────────────────

    title_pct   = min(title_score, 100)
    body_pct    = min(body_score, 100)
    overall_pct = round((title_pct * 0.35) +
                        (body_pct  * 0.65))

    # Quality tier — gives user an immediate label
    if overall_pct >= 85:
        tier         = "Excellent"
        tier_color   = "success"
        tier_message = (
            "This question is well-structured and "
            "contains the information responders need. "
            "You are ready to post."
        )
    elif overall_pct >= 70:
        tier         = "Good"
        tier_color   = "success"
        tier_message = (
            "This question is solid. A few small "
            "improvements could increase your chance "
            "of a great answer."
        )
    elif overall_pct >= 50:
        tier         = "Needs Work"
        tier_color   = "warning"
        tier_message = (
            "This question has some elements but is "
            "missing key information that responders "
            "need. Address the warnings below before "
            "posting."
        )
    else:
        tier         = "Poor"
        tier_color   = "error"
        tier_message = (
            "This question is likely to be closed or "
            "ignored. It is missing several critical "
            "components. Significant revision is needed "
            "before posting."
        )

    return {
        'overall_score': overall_pct,
        'title_score':   title_pct,
        'body_score':    body_pct,
        'tier':          tier,
        'tier_color':    tier_color,
        'tier_message':  tier_message,
        'warnings':      warnings,
        'feedback':      feedback,
        'features':      f
    }