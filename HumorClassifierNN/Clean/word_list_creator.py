from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
import numpy as np


def parse_joke(joke):
    """
    Keeps wordnet nouns from the input, removes everything else
    :param joke: a possible joke (str)
    :returns nouns str: all words listed as nouns by wordnet, separated by spaces
    """
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
    """
    Finds path, wup, and lch similarities
    :param word_list: a list of nouns ([str, str, ...])
    :returns (path_similarity, wup_similarity, lch_similarity) tuple(2d NumPy arrays):
        minimum similarities of each noun to each other noun
    """
    list_len = len(word_list)
    path_similarity = np.zeros((list_len, list_len), np.float32)
    wup_similarity = np.zeros((list_len, list_len), np.float32)
    lch_similarity = np.zeros((list_len, list_len), np.float32)
    for i in range(len(word_list)):
        row_synset = wn.synsets(word_list[i], pos=wn.NOUN)
        for j in range(len(word_list)):
            similarity = word_list[j]
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
                path_similarity[i, j] = min(path_diffs)
                wup_similarity[i, j] = min(wup_diffs)
                lch_similarity[i, j] = min(lch_diffs)
    return path_similarity, wup_similarity, lch_similarity


def get_stats(joke):
    """
    Generates histograms of similarities between nouns in a possible joke
    :param joke: a possible joke (str)
    :returns stats, joke: the histogram and the original joke (tuple(list[NumPy array, NumPy array, NumPy array], str)
    """
    parsed_joke = parse_joke(joke)
    similarities = get_similarities(parsed_joke)
    stats = []
    for similarity in similarities:
        hist = np.nan_to_num(np.histogram(similarity, bins=3, range=(0, 1), density=True)[0]).tolist()
        stats.append(hist)
    return stats, joke

