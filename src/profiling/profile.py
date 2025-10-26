import pandas as pd

csv_path = "Data/Data_Entry_2017.csv"
df = pd.read_csv(csv_path)
print(df.shape)
print(df.columns)
from collections import Counter
all_labels = df['Finding Labels'].str.split('|').explode()
print(all_labels.value_counts().head(20))


