from nltk.stem import PorterStemmer

def stem_text(text_sample : str) -> list:
    porter = PorterStemmer()
    return [porter.stem(i) for i in text_sample.split() ]