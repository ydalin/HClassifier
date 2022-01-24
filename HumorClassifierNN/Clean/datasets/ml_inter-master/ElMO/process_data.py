import random
import pandas as pd

df = pd.read_csv('puns_pos_neg_data.csv', header=None)
ds = df.sample(frac=1)
first_column = df.columns[1]
df = df.drop([first_column], axis=1)
ds.to_csv('positive_data_file.csv')
