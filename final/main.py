from simple_generator import generate_jokes
import keras
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
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
    parsed_joke = parse_joke(joke)
    similarities = get_similarities(parsed_joke)
    stats = []
    for similarity in similarities:
        hist = np.nan_to_num(np.histogram(similarity, bins=3, range=(0, 1), density=True)[0]).tolist()
        stats.append(hist)
    return stats

def get_joke(input='1'):
    jokes = generate_jokes()
    if input == '1':
        present_gathered = 0
        print(str(0) + '% Data processed')
        stats = []
        for i in range(len(jokes)):
            if (i * 100 / len(jokes)) - present_gathered >= 1:
                present_gathered = int(i * 100 / len(jokes))
                print(str(int(i * 100 / len(jokes))) + '% Data processed')
            stats.append(get_stats(jokes[i]))
        model = keras.models.load_model(r"stats_model")
        predictions = model.predict(stats)
        predictions = np.mean(predictions, axis=1)
        print(jokes)
        print(predictions)
        print(np.where(predictions==np.max(predictions))[0][0])
        final_prediction = jokes[np.where(predictions==np.max(predictions))[0][0]]
        return final_prediction

joke = get_joke()
print(joke)
