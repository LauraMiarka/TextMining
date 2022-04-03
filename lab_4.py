import re
import pandas as pd
import matplotlib.pyplot

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from tqdm import tqdm


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


tekscik = pd.read_csv(r'C:\Users\Laura\OneDrive\Pulpit\textmining\Fake.csv')


def main():
    vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)
    x_transform = vectorizer.fit_transform(tekscik['title'][:3])
    print(x_transform.toarray())
