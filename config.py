# Cloud Setup
REGION = 'asia-east1'
PROJECT = 'tensorflow-boilerplate'
BUCKET = PROJECT
TFVERSION = '1.2.0'

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
    'text',  # features
    'spam',  # label
    # passthrough
]  # includes passthrough and label cols if applicable
NUMERIC_COLS = []  # includes passthrough and label cols if applicable
TOKENIZE_COL = 'text'
NGRAM_RANGE = (1, 3)
MAX_TOKENS = 2000
MAX_SEQ_LEN = 200
LABEL_COL = 'spam'
PASSTHROUGH_COLS = []