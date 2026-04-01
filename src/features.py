import re

# ─────────────────────────────────────────────
# UTILITY — strips HTML tags for clean text matching
# This is critical because HQ/LQ_CLOSE are HTML
# but we need clean text for regex pattern matching
# ─────────────────────────────────────────────

def clean_html(text):
    """Remove HTML tags and decode common entities."""
    text = str(text)
    text = re.sub(r'<.*?>', ' ', text)      # strip tags
    text = text.replace('&lt;', '<')         # decode entities
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = re.sub(r'\s+', ' ', text).strip() # normalize whitespace
    return text


# ─────────────────────────────────────────────
# TITLE FEATURES
# ─────────────────────────────────────────────

def title_word_count(title):
    """
    Measures title length in words.
    Too short means vague. Sweet spot is 6-15 words.
    """
    return len(str(title).split())


def title_has_technology(title):
    """
    Detects if title mentions a specific technology.
    A good title names exactly what you are working with
    so the right people find and answer your question.
    """
    technologies = [
        'python', 'java', 'javascript', 'typescript',
        'c++', 'c#', 'golang', 'rust', 'swift', 'kotlin',
        'react', 'angular', 'vue', 'node', 'express',
        'django', 'flask', 'fastapi', 'spring', 'laravel',
        'sql', 'mysql', 'postgresql', 'mongodb', 'sqlite',
        'pandas', 'numpy', 'tensorflow', 'pytorch', 'keras',
        'html', 'css', 'php', 'ruby', 'rails', 'scala',
        'docker', 'kubernetes', 'aws', 'git', 'linux',
        'r ', 'bash', 'powershell', 'jquery', 'bootstrap','react'
    ]
    title_lower = str(title).lower()
    return 1 if any(tech in title_lower
                    for tech in technologies) else 0


def title_has_error_type(title):
    """
    Detects if title mentions a specific error name.
    'TypeError in pandas' is immediately answerable.
    'Python not working' tells responders nothing.
    """
    pattern = (r'\b(TypeError|ValueError|AttributeError|'
               r'KeyError|IndexError|NameError|Exception|'
               r'SyntaxError|RuntimeError|ImportError|'
               r'NullPointerException|404|500|403|'
               r'OverflowError|ZeroDivisionError|'
               r'FileNotFoundError|OSError|IOError)\b')
    return 1 if re.search(pattern, str(title),
                          re.IGNORECASE) else 0


def title_is_vague(title):
    """
    Detects vague complaint words in title.
    These signal the asker hasn't defined their problem.
    Returns 1 if vague — this is a negative signal.
    """
    vague_words = [
        'help', 'issue', 'problem', 'not working',
        'broken', 'weird', 'strange', 'please',
        'urgent', 'stuck', "doesn't work", 'need help',
        'any help', 'how to fix', 'why is'
    ]
    title_lower = str(title).lower()
    return 1 if any(w in title_lower
                    for w in vague_words) else 0


# ─────────────────────────────────────────────
# BODY STRUCTURAL FEATURES
# ─────────────────────────────────────────────

def has_code_block(body):
    """
    Detects code in both HTML and plain text formats.
    HQ and LQ_CLOSE use HTML tags.
    Some questions use markdown backticks or indentation.
    Covering all formats fixes the LQ_EDIT detection gap.
    """
    body_str = str(body)

    # HTML format — primary format in this dataset
    if '<code>' in body_str or '<pre>' in body_str:
        return 1

    # Markdown format
    if '```' in body_str:
        return 1

    # Indented code block (4 spaces or tab at line start)
    if re.search(r'\n(    |\t)\S', body_str):
        return 1

    return 0


def body_word_count(body):
    """
    Word count after stripping HTML.
    Weak signal alone but useful in combination.
    """
    return len(clean_html(body).split())


def body_length_adequate(body):
    """
    Checks if body is in the quality sweet spot.
    Under 50 words = underspecified.
    Over 600 words = unfocused wall of text.
    Sweet spot confirmed by EDA on this dataset.
    """
    count = body_word_count(body)
    return 1 if 50 <= count <= 600 else 0


def code_to_text_ratio(body):
    """
    Ratio of code content to total body length.
    A technically specific question has proportionally
    more code than prose — this captures that signal
    in a way raw length cannot.
    """
    body_str = str(body)
    code_blocks = re.findall(r'<code>(.*?)</code>',
                             body_str, re.DOTALL)
    # Also catch markdown code blocks
    code_blocks += re.findall(r'```.*?```',
                              body_str, re.DOTALL)
    code_length = sum(len(c) for c in code_blocks)
    total_length = len(body_str)
    return round(code_length / total_length, 3) \
           if total_length > 0 else 0.0


# ─────────────────────────────────────────────
# BODY CONTENT FEATURES
# These use clean_html() to strip tags before
# matching so they work on both HTML and plain text
# ─────────────────────────────────────────────

def has_error_keywords(body):
    """
    Layer 1 — Detects explicit technical error terms.
    Uses cleaned text so it works on all body formats.
    """
    clean = clean_html(body).lower()
    patterns = [
        r'typeerror', r'valueerror', r'attributeerror',
        r'keyerror', r'indexerror', r'nameerror',
        r'syntaxerror', r'runtimeerror', r'importerror',
        r'nullpointerexception', r'traceback',
        r'stack trace', r'exception occurred',
        r'error\s*(message|occurs|says|shows)',
        r'(getting|got|throws?|raised?)\s+an?\s+error',
        r'error:',
    ]
    return 1 if any(re.search(p, clean)
                    for p in patterns) else 0


def has_colloquial_error(body):
    """
    Layer 2 — Informal error descriptions.
    Catches natural language like 'I am getting an error'
    even when no specific error type is named.
    """
    clean = clean_html(body).lower()
    patterns = [
        r"i('m| am) (getting|facing|experiencing)",
        r'running into',
        r'throws? an error',
        r'gives? (me )?(an )?error',
        r"(doesn't|does not|won't|will not) work",
        r'(failing|fails)',
        r'not working',
    ]
    return 1 if any(re.search(p, clean)
                    for p in patterns) else 0


def has_attempt_signal(body):
    """
    Layer 3 — Evidence that the user tried something.
    This is the most important signal for avoiding
    'do my work for me' closures on Stack Overflow.
    Broadened patterns catch more natural phrasings.
    """
    clean = clean_html(body).lower()
    patterns = [
        r"i('ve| have)? tried",
        r'i attempted',
        r'already tried',
        r'i was trying',
        r'i thought',
        r"but (it |this )?(doesn't|does not|won't|isn't)",
        r'however',
        r'unfortunately',
        r'i (searched|looked|checked|read|followed)',
        r'according to',
        r'based on',
        r'using the (docs|documentation|tutorial)',
        r'i (used|tested|implemented|wrote)',
    ]
    return 1 if any(re.search(p, clean)
                    for p in patterns) else 0


def has_expected_vs_actual(body):
    """
    Layer 4 — User defined expected vs actual outcome.
    This transforms a vague complaint into a precise,
    answerable question. Strong discriminative signal.
    """
    clean = clean_html(body).lower()
    patterns = [
        r'expected',
        r'supposed to',
        r'should (return|output|work|give|print|show)',
        r'but (got|get|getting|returns?|gives?)',
        r'instead (of|it)',
        r'actually (returns?|outputs?|gives?)',
        r'want(ed)?.{0,30}but',
        r'the (output|result|value|response) (is|was)',
        r'(works|worked) (fine|correctly|well) but',
    ]
    return 1 if any(re.search(p, clean)
                    for p in patterns) else 0


def has_question_mark(body):
    """
    Checks if body contains an actual question.
    Many LQ posts are complaints or demands, not questions.
    Presence of ? signals the asker wants a specific answer.
    """
    return 1 if '?' in str(body) else 0

def detect_question_archetype(title, body):
    clean = clean_html(body).lower()
    title_lower = str(title).lower()
    combined = clean + ' ' + title_lower

    # Explicit signal
    if "no error" in combined or "without error" in combined:
        return 'behavioral'

    debugging_signals = [
        r'error:', r'traceback', r'exception',
        r'crash(es|ing)?', r'throws?',
        r'stack trace',
    ]

    behavioral_signals = [
        r'not updating', r'not working as expected',
        r'unexpected(ly)?', r'strange behavior',
        r'wrong value', r'stale value',
        r'(doesn\'t|does not) (update|change|reflect)',
        r'async(hronous)?', r'timing',
        r'still (shows?|uses?|returns?) (the )?(old|previous|wrong)',
    ]

    howto_signals = [
        r'how (do|can|should) (i|we)',
        r'what is the (best|right|correct) way',
        r'how to', r'is it possible to',
        r'what (should|would) (i|you)',
    ]

    conceptual_signals = [
        r'why (does|is|do|would|can)',
        r'what (does|is|are|happens)',
        r'what happens when',
        r'is this (expected|normal|correct)',
        r'i don\'t understand',
        r'can (you )?explain',
        r'difference between',
        r'when (should|would|do)',
    ]

    if any(re.search(p, combined) for p in debugging_signals):
        return 'debugging'

    if any(re.search(p, combined) for p in behavioral_signals):
        return 'behavioral'

    if any(re.search(p, combined) for p in howto_signals):
        return 'howto'

    if any(re.search(p, combined) for p in conceptual_signals):
        return 'conceptual'

    # fallback
    if '?' in combined:
        return 'conceptual'

    return 'unknown'

# ─────────────────────────────────────────────
# METADATA FEATURES
# ─────────────────────────────────────────────

def tag_count(tags):
    """
    Counts number of tags on the question.
    Tags determine which experts see your question.
    Stack Overflow allows 1-5 tags.
    """
    if not tags or str(tags) == 'nan':
        return 0
    return len(re.findall(r'<[^>]+>', str(tags)))


# ─────────────────────────────────────────────
# MASTER FUNCTION
# Runs all features on one question.
# Returns a dict — becomes one row in feature matrix.
# ─────────────────────────────────────────────

def extract_features(title, body, tags):
    archetype = detect_question_archetype(title, body)
    return {
        'title_word_count':       title_word_count(title),
        'title_has_technology':   title_has_technology(title),
        'title_has_error_type':   title_has_error_type(title),
        'title_is_vague':         title_is_vague(title),
        'has_code_block':         has_code_block(body),
        'body_word_count':        body_word_count(body),
        'body_length_adequate':   body_length_adequate(body),
        'code_to_text_ratio':     code_to_text_ratio(body),
        'has_error_keywords':     has_error_keywords(body),
        'has_colloquial_error':   has_colloquial_error(body),
        'has_attempt_signal':     has_attempt_signal(body),
        'has_expected_vs_actual': has_expected_vs_actual(body),
        'has_question_mark':      has_question_mark(body),
        'tag_count':              tag_count(tags),
        'question_archetype':     archetype,
    }