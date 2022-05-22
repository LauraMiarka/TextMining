import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from tqdm import tqdm

from clean import text_tokenizer, bag_of_words, stem_text


alexa_data = pd.read_csv('C:/Users/Laura/TextMining/alexa_reviews.csv',
                         sep=";",
                         usecols=['rating', 'verified_reviews'])


alexa_data.loc[alexa_data['rating'] == '1', 'rating',] = 1
alexa_data.loc[alexa_data['rating'] == '2', 'rating',] = 2
alexa_data.loc[alexa_data['rating'] == '3', 'rating',] = 3
alexa_data.loc[alexa_data['rating'] == '4', 'rating',] = 4
alexa_data.loc[alexa_data['rating'] == '5', 'rating',] = 5


string = ""
for i in tqdm(range(len(alexa_data['verified_reviews']))):
    string += alexa_data['verified_reviews'].iloc[i] + " "
len(string)

results_cloud = text_tokenizer(string)

print(bag_of_words(results_cloud))
bow = bag_of_words(results_cloud)

wc = WordCloud()
wc.generate_from_frequencies(bow)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
