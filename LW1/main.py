import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
sns.set(rc={'figure.figsize': (10, 8)})

# Data loading

# Loading dataset into dataframe
df = pd.read_csv('./data/OnlineNewsPopularityReduced.csv', delimiter=',')

# General info about dataset
print(df.info())

# Descriptive info
print(df.describe())

# Visualisation of the most interesting columns
plt.figure(figsize=(12, 6))

plt.hist(df['shares'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of shares')
plt.xlabel('Value')
plt.ylabel('Count')

plt.show()

plt.hist(df['n_tokens_title'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of n_tokens_title')
plt.xlabel('Value')
plt.ylabel('Count')
plt.grid(axis='y')

plt.show()

plt.hist(df['n_tokens_content'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of n_tokens_content')
plt.xlabel('Value')
plt.ylabel('Count')
plt.grid(axis='y')

plt.show()

# Data cleaning & Analyzing relationships

news_count_by_weekdays = {
    'monday': df.loc[df['weekday_is_monday'] == 1].shape[0], 
    'tuesday': df.loc[df['weekday_is_tuesday'] == 1].shape[0], 
    'wednesday': df.loc[df['weekday_is_wednesday'] == 1].shape[0], 
    'thursday': df.loc[df['weekday_is_thursday'] == 1].shape[0], 
    'friday': df.loc[df['weekday_is_friday'] == 1].shape[0], 
    'saturday': df.loc[df['weekday_is_saturday'] == 1].shape[0], 
    'sunday': df.loc[df['weekday_is_sunday'] == 1].shape[0]
}

# Determining the day with the most news count
day_with_the_most_news_count = max(news_count_by_weekdays, key=news_count_by_weekdays.get)
print(f"The highest number of news was on {day_with_the_most_news_count}")

# Determining the day with the least news count
day_with_the_least_news_count = min(news_count_by_weekdays, key=news_count_by_weekdays.get)
print(f"The least number of news was on {day_with_the_least_news_count}")

# Visualization of news distribution by days of the week
days = list(news_count_by_weekdays.keys())
counts = list(news_count_by_weekdays.values())

plt.figure(figsize=(12, 6))
plt.bar(days, counts, color='skyblue')

plt.title('News distribution by days of the week')
plt.xlabel('Days')
plt.ylabel('Counts')

plt.show()

# Visualization of the dependence of shares on n_tokens_title 
sns.scatterplot(data=df, x='n_tokens_title', y='shares')
plt.title('The dependence of shares on n_tokens_title')
plt.xlabel('n_tokens_title')
plt.ylabel('shares')

plt.grid(True, linestyle='--', color='black')
plt.show()

# Analysis of the greater impact on popularity
imgs_correlation = df['num_imgs'].corr(df['shares'])
videos_correlation = df['num_videos'].corr(df['shares'])

if abs(imgs_correlation) > abs(videos_correlation):
    print("images influence > videos influence")
else:
    print("videos influence > images influence")

# The popularity on weekends and weekdays
average_weekend_shares = df.loc[df['is_weekend'] == 1, 'shares'].mean().astype(int)
average_not_weekend_shares = df.loc[df['is_weekend'] == 0, 'shares'].mean().astype(int)

categories = ['Weekend', 'Not weekend']
averages = [average_weekend_shares, average_not_weekend_shares]

plt.bar(categories, averages, color=['skyblue', 'salmon'])
plt.title('Average Shares: Weekend vs Not Weekend')
plt.xlabel('Average Shares')
plt.ylabel('Category')
for i, value in enumerate(averages):
    plt.text(i, value, str(value), ha='center', va='bottom')
plt.show()
print(average_weekend_shares > average_not_weekend_shares)

# Dependency between word count and shares
n_tokens_content_shares = df[['n_tokens_content', 'shares']]
plt.scatter(n_tokens_content_shares['n_tokens_content'], n_tokens_content_shares['shares'], alpha=0.5)
plt.title('Dependency between n_token_content and shares')
plt.xlabel('n_tokens_content')
plt.ylabel('Shares')
plt.grid(True)
plt.show()

# Creative assignment

# Comparison of popularity by data_channel
shares_by_data_channels = {
    'lifestyle': df.loc[df['data_channel_is_lifestyle'] == 1, 'shares'].mean().astype(int),
    'entertainment': df.loc[df['data_channel_is_entertainment'] == 1, 'shares'].mean().astype(int),
    'bus': df.loc[df['data_channel_is_bus'] == 1, 'shares'].mean().astype(int),
    'socmed': df.loc[df['data_channel_is_socmed'] == 1, 'shares'].mean().astype(int),
    'tech': df.loc[df['data_channel_is_tech'] == 1, 'shares'].mean().astype(int),
    'world': df.loc[df['data_channel_is_world'] == 1, 'shares'].mean().astype(int)
}

channels = list(shares_by_data_channels.keys())
shares = list(shares_by_data_channels.values())

plt.figure(figsize=(12, 6))
plt.bar(channels, shares, color='skyblue')

plt.title('Average shares by data channels')
plt.xlabel('Channel')
plt.ylabel('Shares')

plt.show()