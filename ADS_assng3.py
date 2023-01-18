import pandas as pd
import numpy as np

pd_df = pd.read_csv("data.csv", skiprows=3)
print('world bank', pd_df)
