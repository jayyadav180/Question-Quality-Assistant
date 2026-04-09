# Question Quality Assistant

A machine learning system that evaluates the quality
of a Stack Overflow question before it is posted
and tells you exactly what to fix.

## Live Demo
Coming soon — deploying to Hugging Face Spaces

---

## Why I Built This

A large percentage of Stack Overflow questions never
get answered. Not because the problem is hard, but
because the question is missing something basic: a
code example, an error message, a clear description
of what went wrong.

The frustrating part is that this feedback only comes
after posting. By then the question has already been
downvoted or closed.

I wanted to build something that catches these problems
before posting. A system that reads a draft question
and tells you specifically what is missing and why
it matters.

---

## What It Does

You paste a draft question including title, body,
and tags. The system analyzes it and returns:

- An overall quality score from 0 to 100 percent
- Separate scores for your title and body
- A quality tier: Excellent, Good, Needs Work, or Poor
- Specific actionable feedback explaining what is
  missing and how to fix it
- An ML-based answerability prediction trained on
  30,000 real Stack Overflow questions

---

## How It Works

The system has three independent layers that work
together.

**Layer 1 - Feature Extraction**
Fourteen structural signals are extracted from the
question using regex and text parsing. These include
whether a code block is present, whether an error
message was included, whether the user described
what they tried, whether the title names a specific
technology, and whether the body contains an actual
question.

**Layer 2 - ML Classification**
A Logistic Regression classifier trained on 30,000
Stack Overflow questions predicts the probability
that the question will be answered. It learned which
combinations of structural features distinguish high
quality questions from those that get closed.

**Layer 3 - Rule-Based Feedback**
A separate rule-based system evaluates the same
features and generates human-readable suggestions.
It first detects what type of question is being
asked: debugging, behavioral, how-to, or conceptual.
It then applies appropriate rules for that type.

A debugging question is evaluated for the presence
of a stack trace. A behavioral question is not,
because behavioral questions rarely produce stack
traces. Applying the same rules to both would
penalize well-written questions for the wrong
reasons.

The ML layer and the rule-based layer are
deliberately kept separate. The model gives a
probability. The rules give an explanation.
Together they form a complete assessment.

---

## Technical Decisions Worth Explaining

**Why I excluded user reputation as a feature**

Stack Overflow questions from high-reputation users
tend to get answered regardless of quality. The
community extends them goodwill, engages with their
posts faster, and is less likely to close their
questions even when those questions are structurally
weak.

This creates a problem for any model trained on
Stack Overflow outcomes. If reputation is included
as a feature, the model learns to predict community
behavior rather than genuine question quality. It
would score a vague question highly simply because
a well-known user wrote it.

The dataset I used did not include a reputation
column, which in hindsight was fortunate. It forced
the model to learn from question structure alone,
which is exactly what a pre-submission assistant
should evaluate. Even if reputation data had been
available, I would have excluded it deliberately.
The system is designed to judge the question, not
the person who wrote it.

**Why I dropped LQ_EDIT questions from training**

The dataset has three quality labels: HQ, LQ_CLOSE,
and LQ_EDIT. When I examined LQ_EDIT questions
directly, I found their body text was stored as
plain text while HQ and LQ_CLOSE bodies were stored
as HTML. This meant every HTML-based feature
extractor produced systematically wrong values for
LQ_EDIT questions.

Training on corrupted feature values would have
introduced noise that no amount of tuning could
fix cleanly. I dropped this class entirely and
converted the problem to binary classification:
high quality versus likely to be closed. This is
documented as a known limitation.

**Why the feedback layer detects question archetype**

Early testing showed that applying the same scoring
rules to all question types produced wrong results.
A behavioral question about React state not updating
was being penalized for missing a stack trace it
would never have. The question was genuinely
well-written but the system scored it poorly.

Adding archetype detection fixed this. The system
now identifies the question type first and applies
appropriate rules for that type. This did not
require retraining the model or changing the feature
set. It only required making the rule-based layer
smarter about what it was looking at.

**Why I chose Logistic Regression over Random Forest**

Both were trained and compared. Logistic Regression
achieved 66.4% accuracy versus Random Forest at
63.5%. More importantly, Logistic Regression with
StandardScaler produced well-calibrated probability
outputs that map cleanly to a 0 to 100 score.
The model is also simpler to explain, which matters
for a system where transparency is part of the value.

---

## Model Performance
```
Dataset:    30,000 Stack Overflow questions
            (15,000 HQ, 15,000 LQ_CLOSE)
Split:      80% train, 20% test
Model:      Logistic Regression + StandardScaler
Accuracy:   66.4%
Baseline:   50% (random on balanced binary problem)
```

66% on a balanced binary problem with known label 
noise from reputation bias is a reasonable result 
for structural features alone. The remaining error 
is partly irreducible community moderation 
decisions are influenced by timing, reputation, 
and visibility that no structural feature can 
capture. 
A v2 with TF-IDF or sentence embeddings 
would likely push accuracy toward 72-75%.

---

## Project Structure
```
question-quality-assistant/
│
├── src/
│   ├── features.py     feature extraction functions
│   ├── model.py        training, evaluation, prediction
│   └── feedback.py     archetype detection and scoring
│
├── app/
│   └── app.py          streamlit interface
│
├── models/
│   ├── classifier.pkl          trained model
│   └── feature_columns.pkl     feature order for prediction
│
├── notebooks/
│   └── 01_EDA.ipynb    exploratory data analysis
│
├── data/               dataset goes here (not committed)
├── requirements.txt
└── README.md
```
---

## What I Would Do Differently In V2

The current system evaluates structure but not 
meaning. Two questions can have identical structural 
features both have code blocks, both have error 
messages but one is a precise, well-formed question 
and the other is incoherent. Structural features 
cannot distinguish between them.

A second version would add TF-IDF vectorization 
or sentence embeddings to capture semantic quality 
alongside structural quality. It would also explore 
re-including LQ_EDIT questions after normalizing 
the body text format, which would give the model 
a richer picture of the middle ground between 
excellent and closed questions.

---

## Dataset

Kaggle — 60k Stack Overflow Questions with Quality Rating  
https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate

Licensed under CC BY-SA 4.0

---

## Built With

Python 3.13 · Scikit-learn · Streamlit · 
Pandas · NumPy · NLTK · Joblib