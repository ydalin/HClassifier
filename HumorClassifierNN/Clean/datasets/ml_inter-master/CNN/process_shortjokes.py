import pandas as pd
import numpy as np

df = pd.read_csv('shortjokes.csv', header=None)
# ds = df.sample(frac=1)
# first_column = df.iloc[:, 1]
# df = df.drop([first_column], axis=1)
print(df.iloc[1:, 1])
df.iloc[1:, 1].to_csv('positive_data_file.csv', index=False, header=False)