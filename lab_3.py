import re
import pandas as pd
import matplotlib.pyplot

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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


text = 'Lorem ipsum dolor :)        sit amet, consectetur;' \
       ' adipiscing elit. Sed eget mattis sem. ;) Mauris ;(' \
       ' egestas erat quam, :< ut     faucibus eros congue :> et.' \
       ' In blandit, mi eu    porta;lobortis, tortor :-)' \
       ' nisl facilisis leo, at ;< tristique augue risus eu risus ;-).'

result = cleanup(text)

print(text)
print(result)
print(type(result))

text1 = "America like South Africa is a" \
        " traumatised sick country - in different ways" \
        " of course - but still messed up."
stop_words = stopwords.words("english")


def text_preproc(x):
    x = ' '.join([word for word in x.split(' ') if word not in stop_words])
    return x


print(text1)
print(text_preproc(text1))

my_lines_list = 'Data science is an interdisciplinary' \
                ' field that uses scientific methods, processes,' \
                ' algorithms and systems to extract knowledge and insights' \
                ' from data in various forms, both structured and unstructured,[1][2]' \
                ' similar to data mining. Data science is a "concept to unify statistics,' \
                ' data analysis, machine learning and their related methods" in order to' \
                ' "understand and analyze actual phenomena" with data.[3]' \
                ' It employs techniques and theories drawn from many fields' \
                ' within the context of mathematics, statistics, information science,' \
                ' and computer science. Turing award winner Jim Gray imagined data' \
                ' science as a "fourth paradigm" of science (empirical, theoretical,' \
                ' computational and now data-driven) and asserted that "everything about' \
                'science is changing because of the impact of information technology" ' \
                'and the data deluge. Data Science is now often used interchangeably with' \
                ' earlier concepts like business analytics,[7] business intelligence,' \
                ' predictive modeling, and statistics. In many cases, earlier approaches' \
                ' and solutions are now simply rebranded as "data science" to be more' \
                ' attractive, which can cause the term to become "dilute[d] beyond usefulness.' \
                '"While many university programs now offer a data science degree, there exists' \
                ' no consensus on a definition or suitable curriculum contents.To its discredit,' \
                ' however, many data-science and big-data projects fail to deliver useful results,' \
                ' often as a result of poor management and utilization of resources. '


def stem_text(text_sample: str) -> list:
    porter = PorterStemmer()
    return [porter.stem(i) for i in text_sample.split()]


print(stem_text(my_lines_list))


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
