from nltk.coconrpus import wordnet as wn
import pandas as pd
import pyjokes




def get_similarities():
    word_list = []
    count = 10
    words = wn.words()
    count_2 = 0
    for word in words:
        if count_2 < 100:
            count_2 += 1
        else:
            syn = wn.synsets(word)[0]
            if syn.pos() == 'n':
                word_list.append(word)
                count -= 1
                if count == 0:
                    break

    path_similarity = pd.DataFrame(index=word_list, columns=word_list)
    wup_similarity = pd.DataFrame(index=word_list, columns=word_list)
    lch_similarity = pd.DataFrame(index=word_list, columns=word_list)
    for i in range(len(path_similarity.index.values)):
        row_synset = wn.synsets(path_similarity.index.values[i], pos=wn.NOUN)
        for j in range(len(path_similarity.columns.values)):
            column_synset = wn.synsets(path_similarity.columns.values[j])
            path_similarity.iloc[i].iloc[j] = row_synset[0].path_similarity(column_synset[0])
            wup_similarity.iloc[i].iloc[j] = row_synset[0].wup_similarity(column_synset[0])
            lch_similarity.iloc[i].iloc[j] = row_synset[0].lch_similarity(column_synset[0])
    return path_similarity, wup_similarity, lch_similarity


similarities = get_similarities()

for similarity in similarities:
    print(similarity)
    print("\n")
