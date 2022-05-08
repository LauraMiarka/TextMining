import re
from nltk.corpus import stopwords

def remove_stop_words(text_sample : str) -> str:
    return [x for x in text_sample.split() if x not in stopwords.words('English')]



def text_cleaner( text_1 : str) -> str:
    #To lower case
    text_1 = text_1.lower()
    #Remove numbers
    text_1 = re.sub('\d ','',text_1)
    #Remove HTML markers
    text_1 = re.sub('<[^>]*>', '', text_1)
    #Remove Interpunction
    text_1 = re.sub('[^a-zA-Z^ ]',' ', text_1)
    #Remove exccess spaces
    text_1 = re.sub(r'[ ]{1,}',' ',text_1)

    return ' '.join(remove_stop_words(text_1))