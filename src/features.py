import re

# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def clean_html(text):
    text = str(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────
# TITLE FEATURES
# ─────────────────────────────────────────────

def title_word_count(title):
    return len(str(title).split())


def title_has_technology(title):
    technologies = [
        'python', 'java', 'javascript', 'typescript',
        'c++', 'c#', 'golang', 'rust', 'swift', 'kotlin',
        'react', 'angular', 'vue', 'node', 'express',
        'django', 'flask', 'fastapi', 'spring', 'laravel',
        'sql', 'mysql', 'postgresql', 'mongodb', 'sqlite',
        'pandas', 'numpy', 'tensorflow', 'pytorch', 'keras',
        'html', 'css', 'php', 'ruby', 'rails', 'scala',
        'docker', 'kubernetes', 'aws', 'git', 'linux',
        'bash', 'powershell', 'jquery', 'bootstrap'
    ]
    title_lower = str(title).lower()
    return 1 if any(tech in title_lower
                    for tech in technologies) else 0


def title_has_error_type(title):
    pattern = (r'\b(TypeError|ValueError|AttributeError|'
               r'KeyError|IndexError|NameError|Exception|'
               r'SyntaxError|RuntimeError|ImportError|'
               r'NullPointerException|404|500|403|'
               r'OverflowError|ZeroDivisionError|'
               r'FileNotFoundError|OSError|IOError|'
               r'leak|segfault|deadlock|race condition)\b')
    return 1 if re.search(pattern, str(title),
                          re.IGNORECASE) else 0


def title_is_vague(title):
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
    body_str = str(body)

    if '<code>' in body_str or '<pre>' in body_str:
        return 1

    # Triple backtick only — single backtick excluded
    # because it appears in prose as inline variable names
    if '```' in body_str:
        return 1

    if re.search(r'\n(    |\t)\S', body_str):
        return 1

    # Unformatted code detection — needs 2+ signals
    unformatted_signals = [
        r'\bimport\s+\w+',
        r'\bfrom\s+\w+\s+import\b',
        r'def\s+\w+\s*\(',
        r'class\s+\w+[\s:(]',
        r'\w+\s*=\s*\w+\s*\(',
        r'if\s+.+:\s*$',
        r'for\s+\w+\s+in\s+',
        r'print\s*\(',
        r'console\.log\s*\(',
        r'df\[.+\]',
        r'SELECT\s+.+FROM\s+',
        r'\w+\.\w+\s*\(',
    ]

    matches = sum(
        1 for p in unformatted_signals
        if re.search(p, body_str,
                     re.MULTILINE | re.IGNORECASE)
    )
    return 1 if matches >= 2 else 0


def body_word_count(body):
    return len(clean_html(body).split())


def body_length_adequate(body):
    count = body_word_count(body)
    return 1 if 50 <= count <= 600 else 0


def code_to_text_ratio(body):
    body_str = str(body)
    code_blocks = re.findall(r'<code>(.*?)</code>',
                             body_str, re.DOTALL)
    code_blocks += re.findall(r'```.*?```',
                              body_str, re.DOTALL)
    code_length = sum(len(c) for c in code_blocks)
    total_length = len(body_str)
    return round(code_length / total_length, 3) \
           if total_length > 0 else 0.0


# ─────────────────────────────────────────────
# BODY CONTENT FEATURES
# ─────────────────────────────────────────────

def has_error_keywords(body):
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
    clean = clean_html(body).lower()
    patterns = [
        r'expected',
        r'supposed to',
        r'should (return|output|work|give|print|show|update|change)',
        r'but (got|get|getting|returns?|gives?|shows?|logs?)',
        r'instead (of|it)',
        r'actually (returns?|outputs?|gives?|shows?)',
        r'want(ed)?.{0,30}but',
        r'the (output|result|value|response|count) (is|was)',
        r'(works|worked) (fine|correctly|well) but',
        r'still (uses?|shows?|logs?|returns?) (the )?(old|previous|wrong|stale)',
        r'(old|previous|stale|wrong) (value|state|data)',
        r'i thought .{0,50} (would|should|will)',
        r'assumed .{0,30} (would|should)',
        r'(doesn\'t|does not) (reflect|update|change)',
        r'not (getting|seeing|receiving) the (new|updated|latest|current)',
    ]
    return 1 if any(re.search(p, clean)
                    for p in patterns) else 0


def has_question_mark(body):
    return 1 if '?' in str(body) else 0


# ─────────────────────────────────────────────
# ARCHETYPE DETECTION
# Returns question type as string.
# Used by feedback.py ONLY — never by ML model.
# ─────────────────────────────────────────────

def detect_question_archetype(title, body):
    """
    Identifies question type to apply appropriate
    scoring rules in feedback.py.
    
    Returns: 'debugging' | 'behavioral' | 
             'howto' | 'conceptual' | 'unknown'
    """
    clean   = clean_html(body).lower()
    t_lower = str(title).lower()
    combined = clean + ' ' + t_lower

    # Explicit no-error signal overrides everything
    if ('no error' in combined or
            'without error' in combined):
        return 'behavioral'

    debugging_signals = [
        r'error:', r'traceback', r'exception',
        r'crash(es|ing)?', r'throws?',
        r'stack trace', r'oserror', r'valueerror',
        r'typeerror', r'syntaxerror',
    ]

    behavioral_signals = [
        r'not updating', r'not working as expected',
        r'unexpected(ly)?', r'strange behavior',
        r'wrong value', r'stale value',
        r'(doesn\'t|does not) (update|change|reflect)',
        r'async(hronous)?', r'timing issue',
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
        r'what (does|is|are|happens) .{0,20} mean',
        r'what happens when',
        r'is this (expected|normal|correct)',
        r'i don\'t understand',
        r'can (you )?explain',
        r'difference between',
        r'when (should|would|do)',
    ]

    if any(re.search(p, combined)
           for p in debugging_signals):
        return 'debugging'

    if any(re.search(p, combined)
           for p in behavioral_signals):
        return 'behavioral'

    if any(re.search(p, combined)
           for p in howto_signals):
        return 'howto'

    if any(re.search(p, combined)
           for p in conceptual_signals):
        return 'conceptual'

    # Honest fallback — unknown is valid
    return 'unknown'


# ─────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────

def tag_count(tags):
    if not tags or str(tags) == 'nan':
        return 0
    return len(re.findall(r'<[^>]+>', str(tags)))


# ─────────────────────────────────────────────
# MASTER FUNCTIONS
# Two versions — one for ML, one for feedback
# ─────────────────────────────────────────────

def extract_features(title, body, tags):
    """
    NUMERIC FEATURES ONLY.
    Used by model.py for training and prediction.
    Archetype excluded — string breaks ML model.
    """
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
    }


def extract_features_with_meta(title, body, tags):
    """
    NUMERIC FEATURES + ARCHETYPE STRING.
    Used by feedback.py and app.py ONLY.
    Never passed to the ML model.
    """
    features = extract_features(title, body, tags)
    features['detected_archetype'] = \
        detect_question_archetype(title, body)
    return features