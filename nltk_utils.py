import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# Natural language processing techniques
# Process involves tokenisation, stemming and then calculate the bag of words, which is used for our training data

# Tokenize sentence entered
def tokenize(sentence):
    # Tokenisation is splitting a string into meaningful units
    return nltk.word_tokenize(sentence) # Using nltk for tokenisation

# Carry out stemming for each word
def stem(word):
    # Stemming is a natural language processing  technique for generating the root form of the words
    # Such as chopping off the end of words
    return stemmer.stem(word.lower())

# Extracting features from words
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    # Loop through all words and if the word from the pattern is included in all words we then put a 1 at that position
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
