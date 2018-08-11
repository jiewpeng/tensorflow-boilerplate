# Cloud Setup
REGION = 'asia-east1'
PROJECT = 'tensorflow-boilerplate'
BUCKET = PROJECT

DELIM = '<SEP>'

RAW_DATA_COLS = [
    'v1',
    'v2'
]
RENAMED_COLS = [
    'spam',
    'text'
]
STRING_COLS = [
    'text', # features
    'spam', # label
    # passthrough
] # includes passthrough and label cols if applicable
NUMERIC_COLS = [] # includes passthrough and label cols if applicable
LABEL_COL = 'spam'
PASSTHROUGH_COLS = []