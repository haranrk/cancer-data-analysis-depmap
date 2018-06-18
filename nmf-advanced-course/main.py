import pandas as pd
import numpy as np

df = pd.read_csv("data/portal-Avana-2018-05-30.csv", index_col=0)
print(df.shape)