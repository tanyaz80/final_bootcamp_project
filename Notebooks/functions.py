
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('tagsets')
nltk.download('omw-1.4')

stop_words = list(stopwords.words('english'))

def clean_up(x):
    """
      
    Cleans up numbers, URLs, and special characters from a string.

    Args:
        x: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    
    #x = str(x).lower().replace("\\","").replace("_"," ")
    #x = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', ' ' , x) 
    #x = re.sub("@[A-Za-z0-9_]+","", x)
    #x = re.sub("#[A-Za-z0-9_]+","", x)

    #x = re.sub(r'\W+',' ',x) # Replace everything non-alpahnumeric by ' '
    #x = re.sub(r'\s+',' ',x) # Replace one or more whitespaces by  ' '
    #x = re.sub(r'\d+',' ',x) # Replace one or more digits by  ' '
    #x = re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)'," ", x) # Replace e-mails by ''
    #x = BeautifulSoup(x, 'lxml').get_text().strip()
    # Replace html tags by ''
    #x = x.replace(' br ',' ')
    
    x = str(x).lower().replace("\\","").replace("_"," ")
    x = re.sub(r'\W+',' ',x) # Replace everything non-alpahnumeric by ' '
    x = re.sub(r'\s+',' ',x) # Replace one or more whitespaces by  ' '
    x = re.sub(r'\d+',' ',x) # Replace one or more digits by  ' '
    x = re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)'," ", x) # Replace e-mails by ''
    # Replace urls by ''
    x = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', ' ' , x) 
    # Replace html tags by ''
    x = BeautifulSoup(x, 'lxml').get_text().strip()
    x = re.sub("#[A-Za-z0-9_]+","", x)
    x = x.replace(' br ',' ')

    return x


    return x

    
def load(filename = "filename.pickle"): 
    try: 
        with open(filename, "rb") as file: 
            return pickle.load(file) 
    except FileNotFoundError: 
        print("File not found!") 
    
def tokenize(s):
    """
    Tokenize a string.

    Args:
        s: String to be tokenized.

    Returns:
        A list of words as the result of tokenization.
    """
    tokens = nltk.word_tokenize(s)
    
    return tokens

def stem_and_lemmatize(l):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    ps = PorterStemmer()
    stemmed_output=([ps.stem(w) for w in l])

    lemmatizer = WordNetLemmatizer()
    stemmed_lemmatized_output = ([lemmatizer.lemmatize(w) for w in stemmed_output])
    

    return stemmed_lemmatized_output

def remove_stopwords(l):
  """
    Remove English stopwords from a list of strings.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """
  tokens_without_sw = [word for word in l if not word in stop_words]

    
  return tokens_without_sw
