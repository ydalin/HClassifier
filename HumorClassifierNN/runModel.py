import numpy as np
from ClassifierModel import ClassifierNNModel as ClassNN
from word_list_creator import results
from gather_data import gather_data
import tensorflow as tf


train_data, test_data, validation_data = gather_data()

# noun_sims, verb_sims, word_list = results


class_model = ClassNN(train_data, test_data, validation_data).get_model()

class_model.compile()
print(class_model.summary())
# class_model.fit(train_data, epochs=150, batch_size=10)