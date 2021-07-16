import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime

df = pd.read_csv("dataset.csv")


# taking only first 5 relevant rows
df = df.iloc[:, :5]
# dropping rows with missing info (2021)
df = df.dropna()
# show only values for all sub categories
df = df[df['AUSPRAEGUNG'] == 'insgesamt']
# remove sum info from the value column
df = df[df['MONAT'] != 'Summe']
# make it into descending order chronologically
df = df[::-1]


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