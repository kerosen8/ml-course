import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
sns.set(rc={'figure.figsize': (10, 8)})

def detect_outliers(data):
    """
    Визначає, чи є викиди даних за допомогою метода межквартільного розмаху (IQR).

    :param data: масив числових даних
    :return: лист індексів значень які викидаються
    """
    data = np.array(data)
    
    # вираховуємо перший та третий квартиль
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    
    # вираховуємо межквартильний розмах 
    IQR = Q3 - Q1
    
    # визначаємо нижню та верхню границі для викидів
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # шукаємо індекси викидів
    outliers_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
    
    return outliers_indices

# target змінна: shares; основна задача прогнозної моделі: регресія

# Data loading

# Загрузка даних у датасет
df = pd.read_csv('./data/OnlineNewsPopularityReduced.csv', delimiter=',')

# Загальна інформація про набір даних: кількість, тип змінних і наявніть нульових значень
print(df.info())
print()

# Описова статистика по числовим змінним (по всім, так як всі змінні - числові)
for column in df.columns:
    print(df[column].describe())

# Візуалізація розподілів найцікавіших змінних

# Візуалізація розподілу змінної shares
df['shares'].hist(bins = 100)
plt.title('Histogram of shares')
plt.xlabel('Value')
plt.ylabel('Count')

plt.show()

shares_outliers = detect_outliers(df['shares'])

print(f"Висновок за розподілом змінної 'shares': змінна має {len(shares_outliers)} викидів.\nНайбільша кількість даних зосереджена в інтервалі значень між 0 та 1000.")
print()

# Візуалізація розподілу змінної n_tokens_title
df['n_tokens_title'].hist(bins = 50)
plt.title('Histogram of n_tokens_title')
plt.xlabel('Value')
plt.ylabel('Count')
plt.grid(axis='y')

plt.show()

n_tokens_title_outliers = detect_outliers(df['n_tokens_title'])

print(f"Висновок за розподілом змінної 'n_token_title': змінна має {len(n_tokens_title_outliers)} викидів.\nРозподіл нормальний.")
print()

# Візуалізація розподілу змінної n_tokens_сontent
df['n_tokens_content'].hist(bins = 50)
plt.title('Histogram of n_tokens_content')
plt.xlabel('Value')
plt.ylabel('Count')
plt.grid(axis='y')

plt.show()

n_tokens_content_outliers = detect_outliers(df['n_tokens_content'])

print(f"Висновок за розподілом змінної 'n_token_count': змінна має {len(n_tokens_content_outliers)} викидів.\nПік по кількості змінних знаходиться на проміжку значень 100-200, далі йде низхідний тренд після піку.")
print()

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

# Визначення дня, в який була опублікована найбільша кількість статей
day_with_the_most_news_count = max(news_count_by_weekdays, key=news_count_by_weekdays.get)
print(f"День, в який було опубліковано найбільше статей: {day_with_the_most_news_count}")

# Визначення дня, в який була опублікована найменша кількість статей
day_with_the_least_news_count = min(news_count_by_weekdays, key=news_count_by_weekdays.get)
print(f"День, в який було опубліковано найменше статей: {day_with_the_least_news_count}")
print()

# Візуалізація розподілу кількості опублікованих статей за днями тижня
days = list(news_count_by_weekdays.keys())
counts = list(news_count_by_weekdays.values())

plt.bar(days, counts, color='skyblue')

plt.title('News distribution by days of the week')
plt.xlabel('Days')
plt.ylabel('Count of news')

plt.show()

# Зв'язок між довжиною заголовку статті та результуючою змінною
n_tokens_title_corr_to_shares = df['n_tokens_title'].corr(df['shares'])
print(f"Кореляція між n_tokens_title та shares: {n_tokens_title_corr_to_shares}.\nТака кореляція вказує на дуже слабкий позитивний зв'язок між цими змінними, тобто довжина заголовку не має суттєвого впливу на популярність статті.")
print()

# Аналіз впливу картинок та відеороликів на популярність статті
imgs_correlation = df['num_imgs'].corr(df['shares'])
videos_correlation = df['num_videos'].corr(df['shares'])

print(f"Кореляція між кількістю картинок в статті та її популярністю: {imgs_correlation}")
print(f"Кореляція між кількістю відеороликів в статті та її популярністю: {videos_correlation}")
print("Кореляція між num_images і shares більша, ніж між num_images і shares, тобто зображення мають більший вплив на популярність статті, однак кореляція між кількістю зображень та shares достатньо мала для того щоб казати що вплив несуттєвий.")
print()

# Порівняння популярності у будні та вихідні
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

# Залежніть популярності від кількості символів в статті
n_tokens_content_shares = df[['n_tokens_content', 'shares']]
plt.scatter(data=df, x='n_tokens_content', y='shares')
plt.title('Dependency between n_token_content and shares')
plt.xlabel('n_tokens_content')
plt.ylabel('Shares')
plt.grid(True)
plt.show()

n_tokens_content_corr_to_shares = df['n_tokens_content'].corr(df['shares'])
print(f"Кореляція між n_tokens_content і shares: {n_tokens_content_corr_to_shares}\nВиходячи з візуалізації та кореляції, яка майже нульова, можна зробити висновок: між кількістю символів в тексті та популярністю є дуже слабкий від'ємний зв'язок, тобто збільшення довжини текста не впливає на популярність статей.") 
print()

# Creative assignment

# Порівняння популярності статей за data_channel
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

plt.bar(channels, shares, color='skyblue')

plt.title('Average shares by data channels')
plt.xlabel('Channel')
plt.ylabel('Shares')

plt.show()

print("Висновок про популярність статей за data_channel: виходячи зі стовпчатої діаграми, можна зрозуміти, що всі категорії мають відносно однакову популярність між читачами.")
print()

# Що більше впливає на shares: rate_positive_words або rate_negative_words?
rate_positive_words_corr_to_shares = df['rate_positive_words'].corr(df['shares'])
rate_negative_words_corr_to_shares = df['rate_negative_words'].corr(df['shares'])

print(f"Кореляція між rate_positive_words і shares: {rate_positive_words_corr_to_shares}.")
print(f"Кореляція між rate_negative_words і shares: {rate_negative_words_corr_to_shares}.")

print("Висновок про вплив позитивних та негативних слів у статті на її популярність: оскільки обидва коефіцієнта кореляції від'ємні та близькі до нуля, це свідчить про те, що за даними датасету як позитивні, так і негативні слова задають тенденцію на спад шерів. Але оскількі коеф. кореляції близький до нуля, то цей вплив є незначним (майже відсутній).")
print()