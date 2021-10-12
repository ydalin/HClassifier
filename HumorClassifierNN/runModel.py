import numpy as np

from keras.datasets import imdb
from ClassifierModel import ClassifierNNModel as ClassNN
word_indices = imdb.get_word_index(path="imdb_word_index.json")

imdb_data = imdb.load_data(num_words=100)

class_model = ClassNN(imdb_data).get_model()
print(class_model)
class_model.train()
