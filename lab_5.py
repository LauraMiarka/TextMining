import re

import numpy as np
import pandas as pd
import matplotlib.pyplot

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def cleanup(tekst: str) -> str:
    x = tekst
    emotki = re.findall('[:;]-?[)(><]', x)
    wynik1 = re.sub('[0-9]', '', x, count=0, flags=0)
    wynik2 = re.sub('(<([^>]+)>.*?)', '', wynik1, count=0, flags=0)
    wynik3 = re.sub('([:;]-?[)(><])', '', wynik2, count=0, flags=0)
    wynik4 = re.sub('[,;:\.]|', '', wynik3, count=0, flags=0)
    wynik5 = re.sub(' +', ' ', wynik4)
    small = wynik5.lower()
    emotki_string = ' '.join([str(element) for element in emotki])
    laczenie_tekstow = small + emotki_string
    return laczenie_tekstow


stop_words = stopwords.words("english")


def text_preproc(x):
    x = ' '.join([word for word in x.split(' ') if word not in stop_words])
    return x


def stem_text(text_sample: str) -> list:
    porter = PorterStemmer()
    return [porter.stem(i) for i in text_sample.split()]


df = pd.read_csv(r"C:\Users\Laura\OneDrive\Pulpit\textmining\True.csv")

string = ""
for i in tqdm(range(len(df['title']))):
    string += df['title'].iloc[i] + " "
len(string)


def bag_of_words(words: list) -> dict:
    bow = {}
    for word in words:
        if word not in bow.keys():
            bow[word] = 1
        else:
            bow[word] += 1
    return bow


string = ""
for i in tqdm(range(len(df['title']))):
    string += df['title'].iloc[i] + " "
len(string)

results_cloud = stem_text(text_preproc(cleanup(string)))

print(bag_of_words(results_cloud))
bow = bag_of_words(results_cloud)

wc = WordCloud()
wc.generate_from_frequencies(bow)
matplotlib.pyplot.imshow(wc, interpolation='bilinear')
matplotlib.pyplot.axis("off")
matplotlib.pyplot.show()


def longer_words(lista: list) -> list:
    new_list = []
    for word in lista:
        if len(word) > 3:
            new_list.append(word)
    return new_list


def text_tokenizer(text):
    tekst = cleanup(text)
    tekst = stem_text(tekst)
    tekst = text_preproc(tekst)
    tekst = longer_words(tekst)
    return tekst


def top_tok(list_of_tokens, token_words, how_many) -> list:
    list_of_tokens = list_of_tokens
    top_list = []
    for i in range(how_many):
        token_index = np.argmax(list_of_tokens)
        top_list.append(token_words[token_index])
        list_of_tokens[token_index] = 0
    return top_list


def top_doc(list_of_documents, how_many) -> list:
    list_of_documents = list_of_documents
    top_list = []
    for i in range(how_many):
        token_index = np.argmax(list_of_documents, 0)
        top_list.append(token_index)
        list_of_documents[token_index] = 0
    return top_list


tekscik = pd.read_csv(r'C:\Users\Laura\OneDrive\Pulpit\textmining\Fake.csv')


def main():
    vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)
    x_transform = vectorizer.fit_transform(tekscik['title'][:3])
    print(x_transform.toarray())

    vectorizer_count = CountVectorizer(tokenizer=text_tokenizer)
    count_transform = vectorizer_count.fit_transform(df['title'])
    vectorizer_tfid = TfidfVectorizer(tokenizer=text_tokenizer)
    tfid_transform = vectorizer_tfid.fit_transform(df['title'])
    # print(count_transform.toarray()) #będą same jedynki brak zer

    top_often_tokens: list = top_tok(count_transform.toarray().sum(axis=0),
                                     vectorizer_count.get_feature_names_out(), 10)
    print("Top 10 tokens(appear)")
    print(top_often_tokens)
    top_important_tokens: list = top_tok(tfid_transform.toarray().sum(axis=0),
                                         vectorizer_tfid.get_feature_names_out(), 10)
    print("Top 10 most important tokens")
    print(top_important_tokens)
    top_often_dokuments: list = top_doc(count_transform.toarray().sum(axis=0), 10)
    print("Top 10 docsow(appear)")
    for doc in top_often_dokuments:
        print(df['title'][doc])
