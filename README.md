# DS-task4

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

data= pd.read_csv('twitter_training.csv')

data.head()

col_names=['ID','Entity','Sentiments','Contest']
df=pd.read_csv('twitter_training.csv', names=col_names)

df.head()
df.shape
df.describe

df.isnull().sum()
df.dropna(axis=0,inplace=True)
df.isnull().sum()

df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()

df.shape
sentiment_counts=df['Sentiments'].value_counts()
sentiment_counts

plt.figure(figsize=(6,3))
sentiment_counts.plot(kind='bar',color=['red','green','blue','yellow'])
plt.title('Sentiment Distribution')
plt.xlabel('Number of Tweets')
plt.xticks(rotation=0)
plt.show()

brand_data=df[df['Entity'].str.contains('Microsoft',case=False)]
brand_sentiment_counts=brand_data['Sentiments'].value_counts()
brand_sentiment_counts

plt.figure(figsize=(6,6))
plt.pie(brand_sentiment_counts,labels=brand_sentiment_counts.index,autopct='%1.11f%%',startangle=140)
plt.show()
