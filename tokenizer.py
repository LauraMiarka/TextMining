from cleaning import text_cleaner
from stemming import stem_text

def text_tokenizer(text):
    text = text_cleaner(text)
    text = stem_text(text)
    return [word for word in text if len(word)>3]
