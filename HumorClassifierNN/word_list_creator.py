import nltk
from nltk.corpus import wordnet as wn
from nltk import sent_tokenize
from nltk import word_tokenize
import pandas as pd
import pyjokes
import numpy as np

joke = pyjokes.get_joke()
print(joke)
print('\n')

def parse_joke(joke):
    nouns = []
    verbs = []
    for word in word_tokenize(joke):
        synset = wn.synsets(word)
        if len(synset) > 0:
            for syn in synset:
                if word not in nouns and syn.pos() == 'n':
                    nouns.append(word)
                if word not in verbs and syn.pos() == 'v':
                    verbs.append(word)
    return nouns, verbs

def get_similarities(word_list, pos):
    path_similarity = pd.DataFrame(index=word_list, columns=word_list)
    wup_similarity = pd.DataFrame(index=word_list, columns=word_list)
    lch_similarity = pd.DataFrame(index=word_list, columns=word_list)
    for i in range(len(path_similarity.index.values)):
        row_synset = wn.synsets(path_similarity.index.values[i], pos=pos)
        for j in range(len(path_similarity.columns.values)):
            column_synset = wn.synsets(path_similarity.columns.values[j], pos=pos)
            path_similarity.iloc[i].iloc[j] = row_synset[0].path_similarity(column_synset[0])
            wup_similarity.iloc[i].iloc[j] = row_synset[0].wup_similarity(column_synset[0])
            lch_similarity.iloc[i].iloc[j] = row_synset[0].lch_similarity(column_synset[0])
    return path_similarity, wup_similarity, lch_similarity



def results(joke):
    """
    param joke: a joke (str)
    returns: tuple:
        (noun similarities: path_similarity, wup_similarity, lch_similarity),
        (verb similarities: path_similarity, wup_similarity, lch_similarity),
        word list
    """
    words = parse_joke(joke)
    similarities_nouns = get_similarities(words[0], wn.NOUN)
    similarities_verbs = get_similarities(words[1], wn.VERB)
    return similarities_nouns, similarities_verbs, words

results = results(joke)
similarities_nouns = results[0]
similarities_verbs = results[1]
word_list = results[2]
print('word list: ')
print(results[2])
print('\npath similarity:')
print(similarities_nouns[0])
print('\nwup similarity:')
print(similarities_nouns[1])
print('\nlch similarity:')
print(similarities_nouns[2])