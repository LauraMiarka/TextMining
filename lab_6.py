import re

import numpy as np
import pandas as pd
import matplotlib.pyplot
from matplotlib import pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from prettytable import PrettyTable


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


def top_tok(list_of_tokens, token_words, how_many) -> dict:
    top_words = []
    top_counts = []
    top_dict = {}
    for i in range(how_many):
        token_index = np.argmax(list_of_tokens)
        top_words.append(token_words[token_index])
        top_counts.append(list_of_tokens[token_index])
        list_of_tokens[token_index] = 0
    for key, value in zip(top_words, top_counts):
        top_dict[key] = value
    return top_dict


def top_doc(list_of_documents, how_many) -> list:
    list_of_documents = list_of_documents
    top_list = []
    for i in range(how_many):
        token_index = np.argmax(list_of_documents, 0)
        top_list.append(token_index)
        list_of_documents[token_index] = 0
    return top_list


def plot_table_most_important(top_dict, title):
    words = list(top_dict.keys())[::-1]
    counts = list(top_dict.values())[::-1]
    plt.subplots(figsize=(11, 5))
    y_pos = np.arange(len(words))
    plt.barh(y_pos, counts)
    plt.yticks(y_pos, words)
    plt.ylabel("Term")
    plt.xlabel("Count")
    plt.title(title)
    plt.show()
    pretty_table = PrettyTable()
    pretty_table.title = title
    pretty_table.add_column("Term", words[::-1])
    pretty_table.add_column("Count", counts[::-1])
    return pretty_table

data_true = pd.read_csv('C:/Users/Laura/OneDrive/Pulpit/textmining/True.csv', usecols=['title', 'text'])
data_fake = pd.read_csv('C:/Users/Laura/OneDrive/Pulpit/textmining/Fake.csv', usecols=['title', 'text'])


def main():
    vectorizer_count = CountVectorizer(tokenizer=text_tokenizer)
    count_transform_true = vectorizer_count.fit_transform(data_true['title'])
    count_transform_fake = vectorizer_count.fit_transform(data_fake['title'])
    vectorizer_tfid = TfidfVectorizer(tokenizer=text_tokenizer)
    tfid_transform_true = vectorizer_tfid.fit_transform(data_true['title'])
    tfid_transform_fake = vectorizer_tfid.fit_transform(data_fake['title'])
    vectorizer_binary = CountVectorizer(tokenizer=text_tokenizer, binary=True)
    binary_transform_true = vectorizer_binary.fit_transform(data_true['title'])
    binary_transform_fake = vectorizer_binary.fit_transform(data_fake['title'])
    print(plot_table_most_important(
        top_tok(count_transform_fake.toarray().sum(axis=0), vectorizer_count.get_feature_names_out(), 15),
        "Występujace tylko w tytułach fałszywych wiadomości"))
    print(plot_table_most_important(
        top_tok(count_transform_true.toarray().sum(axis=0), vectorizer_count.get_feature_names_out(), 15),
        "Występujace tylko w tytułach prawdziwych wiadomości"))
    print(plot_table_most_important(
        top_tok(tfid_transform_true.toarray().sum(axis=0), vectorizer_tfid.get_feature_names_out(), 15),
        "Kluczowe tokeny prawdziwych wiadomości na podstawie miary TF-IDF"))
    print(plot_table_most_important(
        top_tok(binary_transform_true.toarray().sum(axis=0), vectorizer_binary.get_feature_names_out(), 15),
        "Crucial tokens based on binary weight"))
