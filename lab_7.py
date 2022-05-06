import re

import numpy as np
import pandas as pd
import matplotlib.pyplot
from matplotlib import pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics



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


data_true["type"] = 1
data_fake["type"] = 0

df = pd.concat([data_true, data_fake])


def main():
    vectorizer_count = CountVectorizer(tokenizer=text_tokenizer)
    count_transform = vectorizer_count.fit_transform(df['title'])
    x_train, x_test, y_train, y_test = train_test_split(count_transform, df['type'], test_size=0.33, random_state=42)
    clf_tree = DecisionTreeClassifier().fit(x_train, y_train)
    y_pred = clf_tree.predict(x_test)
    print("Decission Tree:", round(metrics.accuracy_score(y_test, y_pred), 2))
    clf_forecast = RandomForestClassifier().fit(x_train, y_train)
    y_pred = clf_forecast.predict(x_test)
    print("Random Forecast:", round(metrics.accuracy_score(y_test, y_pred), 2))
    clf_svm = LinearSVC().fit(x_train, y_train)
    y_pred = clf_svm.predict(x_test)
    print("SVM:", round(metrics.accuracy_score(y_test, y_pred), 2))
    clf_ada = AdaBoostClassifier().fit(x_train, y_train)
    y_pred = clf_ada.predict(x_test)
    print("AdaBoost:", round(metrics.accuracy_score(y_test, y_pred), 2))
    clf_bagging = BaggingClassifier().fit(x_train, y_train)
    y_pred = clf_bagging.predict(x_test)
    print("Bagging:", round(metrics.accuracy_score(y_test, y_pred), 2))


