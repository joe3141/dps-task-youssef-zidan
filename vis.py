import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime

from utils import clean_dataset

df = pd.read_csv("dataset.csv")


df = clean_dataset(df)

# Add datetime field
df['date'] = df['MONAT'].apply(lambda x: x[:4] + "/" + x[4:])
df['date'] = pd.to_datetime(df['date'])


categories = df['MONATSZAHL'].unique()
print(df['MONATSZAHL'].value_counts())


import matplotlib.pyplot as plt

for category in categories:
    
    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(x='date', y='WERT', data=df[df['MONATSZAHL'] == category], 
                 ax=ax, color='mediumblue', label=category)

plt.show()