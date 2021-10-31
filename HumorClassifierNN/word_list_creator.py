import nltk
from nltk.corpus import wordnet as wn
from nltk import sent_tokenize, word_tokenize, pos_tag
import pandas as pd
import pyjokes
import numpy as np
from scipy import stats as scipystats
import csv
import math


def parse_joke(joke):
    nouns = []
    verbs = []
    joke_tagged = pos_tag(word_tokenize(joke))
    tag_filtered = ''
    for word in joke_tagged:
        # if (word[1][0] == 'N' or word[1][0] == 'V') and (word[0] not in tag_filtered):
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
                # elif word not in verbs and syn.pos() == 'v':
                #     verbs.append(word)
                #     break
    return nouns, verbs, joke


# def get_similarities(word_list, pos):
#     path_similarity = pd.DataFrame(index=word_list, columns=word_list)
#     wup_similarity = pd.DataFrame(index=word_list, columns=word_list)
#     lch_similarity = pd.DataFrame(index=word_list, columns=word_list)
#     for i in range(len(path_similarity.index.values)):
#         row_synset = wn.synsets(path_similarity.index.values[i], pos=pos)
#         for j in range(len(path_similarity.columns.values)):
#             column_synset = wn.synsets(path_similarity.columns.values[j], pos=pos)
#             path_similarity.iloc[i].iloc[j] = row_synset[0].path_similarity(column_synset[0])
#             wup_similarity.iloc[i].iloc[j] = row_synset[0].wup_similarity(column_synset[0])
#             lch_similarity.iloc[i].iloc[j] = row_synset[0].lch_similarity(column_synset[0])
#     return path_similarity, wup_similarity, lch_similarity

# Only wup similarities
def get_similarities(word_list, pos):
    path_similarity = pd.DataFrame(index=word_list, columns=word_list)
    wup_similarity = pd.DataFrame(index=word_list, columns=word_list)
    lch_similarity = pd.DataFrame(index=word_list, columns=word_list)
    for i in range(len(wup_similarity.index.to_numpy())):
        row_synset = wn.synsets(wup_similarity.copy().index.values[i], pos=pos)
        for j in range(len(wup_similarity.columns.to_numpy())):
            similarity = wup_similarity.copy().columns.values[j]
            if type(similarity) is str:
                column_synset = wn.synsets(similarity, pos=pos)
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


# def results(joke):
#     """
#     param joke: a joke (str)
#     returns: tuple:
#         (noun similarities: path_similarity, wup_similarity, lch_similarity),
#         (verb similarities: path_similarity, wup_similarity, lch_similarity),
#         word list
#     """
#     words = parse_joke(joke)
#     similarities_nouns = get_similarities(words[0], wn.NOUN)
#     similarities_verbs = get_similarities(words[1], wn.VERB)
#     similarities_all = []
#     for i in range(len(similarities_nouns)):
#         similarities_all.append(pd.concat([similarities_nouns[i], similarities_verbs[i]]))
#     return similarities_nouns, similarities_verbs, similarities_all


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
    # similarities_verbs = get_similarities(words[1], wn.VERB)
    similarities_verbs = -1

    # similarities_all = []
    # for i in range(len(similarities_nouns)):
    #     similarities_all.append(pd.concat([similarities_nouns[i], similarities_verbs[i]]))
    return similarities_nouns, similarities_verbs, words[2]


# def get_stats(joke):
#     result = results(joke)
#     print(result)
#     # Create list of dataframe stats for nouns, verbs, and both: standard dev, max (below 1), min, mean, mode, median
#     stats = []
#     for res in result:
#         for ele in res:
#             ele = ele.fillna(value=-1).replace(1, -1)
#             ele = ele.to_numpy().flatten()
#             ele = ele[ele > -1]
#             if len(ele) == 0:
#                 stats.append(-1)
#                 stats.append(-1)
#                 stats.append(-1)
#                 stats.append(-1)
#                 stats.append(-1)
#             else:
#                 stats.append(ele.max())
#                 stats.append(ele.mean())
#                 stats.append(ele.min())
#                 stats.append(scipystats.mode(ele)[0][0])
#                 stats.append(scipystats.median_abs_deviation(ele))
#     return stats


def get_stats(joke):
    result = results(joke)
    similarities = result[0]
    thresh = 0
    stats = []
    for similarity in similarities:
        similarity[similarity < thresh] = 0
        stats.append(similarity.values.sum())
        # hist = np.nan_to_num(np.histogram(similarity, bins=3, range=(0, 1), density=True)[0]).tolist()
        # if len(hist) == 3 and all(hist) is not None:
        #     stats.append(hist)

    # Create list of dataframe stats for nouns, verbs, and both: standard dev, max (below 1), min, mean, mode, median
    # stats = []
    # for ele in result[:-1]:
    #     ele = ele.fillna(value=-1).replace(1, -1)
    #     ele = ele.to_numpy().flatten()
    #     ele = ele[ele > -1]
    #     if len(ele) == 0:
    #         pass
    #         # stats.append(-1)
    #         # stats.append(-1)
    #         # stats.append(-1)
    #         # stats.append(-1)
    #         # stats.append(-1)
    #     else:
    #         stats.append(ele.max())
    #         # stats.append(ele.mean())
    #         # stats.append(ele.min())
    #         # stats.append(scipystats.mode(ele)[0][0])
    #         # stats.append(scipystats.median_abs_deviation(ele))
    return stats, result[-1]



# pyjoke = pyjokes.get_joke()
# print(pyjoke)
#
# statistics = get_stats(pyjoke)
# print('stats: ' + str(statistics))