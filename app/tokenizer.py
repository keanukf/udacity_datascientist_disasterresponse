import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def tokenize(text):
    """
    Normalize, tokenize, and lemmatize text for downstream ML pipelines.

    Args:
        text (str): Raw user-supplied text.

    Returns:
        list[str]: Clean tokens suitable for vectorization.
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok:
            clean_tokens.append(clean_tok)

    return clean_tokens

