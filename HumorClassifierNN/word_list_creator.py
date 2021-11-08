import nltk
from nltk.corpus import wordnet as wn
from nltk import sent_tokenize, word_tokenize, pos_tag
import pandas as pd
import numpy as np
import csv
import math


def parse_joke(joke):
    nouns = []
    joke_tagged = pos_tag(word_tokenize(joke))
    tag_filtered = ''
    for word in joke_tagged:
        if word[1][0] == 'N' and word[0] not in tag_filtered:
            tag_filtered += ' '
            tag_filtered += word[0]
    for word in word_tokenize(tag_filtered):
        synset = wn.synsets(word)
        if len(synset) > 0:
            for syn in synset:
                if word not in nouns and syn.pos() == 'n':
                    nouns.append(word)
                    break
    return nouns


def get_similarities(word_list):
    path_similarity = pd.DataFrame(index=word_list, columns=word_list)
    wup_similarity = pd.DataFrame(index=word_list, columns=word_list)
    lch_similarity = pd.DataFrame(index=word_list, columns=word_list)
    for i in range(len(wup_similarity.index.to_numpy())):
        row_synset = wn.synsets(wup_similarity.copy().index.values[i], pos=wn.NOUN)
        for j in range(len(wup_similarity.columns.to_numpy())):
            similarity = wup_similarity.copy().columns.values[j]
            if type(similarity) is str:
                column_synset = wn.synsets(similarity, pos=wn.NOUN)
                path_diffs = []
                wup_diffs = []
                lch_diffs = []
                for r in row_synset:
                    for c in column_synset:
                        path_diffs.append(r.path_similarity(c))
                        wup_diffs.append(r.wup_similarity(c))
                        lch_diffs.append(r.lch_similarity(c))
                path_similarity.iloc[i, j] = min(path_diffs)
                wup_similarity.iloc[i, j] = min(wup_diffs)
                lch_similarity.iloc[i, j] = min(lch_diffs)
            else:
                path_similarity.iloc[i, j] = 0.0
                wup_similarity.iloc[i, j] = 0.0
                lch_similarity.iloc[i, j] = 0.0
    return path_similarity, wup_similarity, lch_similarity


def get_stats(joke):
    parsed_joke = parse_joke(joke)
    similarities = get_similarities(parsed_joke)
    thresh = 0.5
    stats = []
    for similarity in similarities:
        similarity[similarity < thresh] = 0
        hist = np.nan_to_num(np.histogram(similarity, bins=3, range=(0, 1), density=True)[0]).tolist()
        if len(hist) == 3 and all(hist) is not None:
            stats.append(hist)
    return stats, joke

