from tensorflow.keras.layers import *
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup


def text_cleaner(text, num):
    """
    This function cleans a text by removing everything :)
    """
    # define new string
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"', '', newString)
    newString = re.sub(r'\*\*|\.\.|:\)|:\(', '', newString)  # Remove *, .., :), and :(
    newString = re.sub(r'\.\.\.', '.', newString)  # Replace ... with .
    newString = re.sub(r'!!!', '!', newString)  # Replace !!! with !
    newString = re.sub(r'!!', '!', newString)  # Replace !! with !
    newString = re.sub(r'[\[\]]', '', newString)  # Remove [ and ]

    tokens = newString.split()

    long_words = []
    for i in tokens:
        if len(i) > 1 or i == 'i':
            long_words.append(i)
    return (" ".join(long_words)).strip()


