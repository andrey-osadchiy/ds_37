Проект
Данные сохранены в файле yandex_music_project.csv в папке /datasets.
Описание колонок:
userID — идентификатор пользователя;
Track — название трека;
artist — имя исполнителя;
genre — название жанра;
City — город пользователя;
time — время начала прослушивания;
Day — день недели.

Чтобы посчитать значения после логической индексации или группировки, добавьте название столбца:
# хуже
df.groupby('сity').count()
df[df['city'] == 'Moscow'].count() 

# подсчёт в датафрейме после очистки от пропусков и группировки
df.groupby('сity').count() 

# выберем столбец genre после группировки
df.groupby('city')['genre'].count() 


В тетрадке Jupyter по-разному выводите на экран разные типы данных:
датафреймы и Series — функцией display();
любые другие данные — функцией print().

df = pd.read_csv('/datasets/yandex_music_project.csv')
number = 42

# хорошо
display(df)
display(df.head())
print(number)

# хуже
print(df)
print(df.head())
display(number)

Детали
Функция display() выведет табличные данные с красивым форматированием. А print() ничего не знает о таблицах и покажет их как обычный текст.



# хорошо
filtered_df = df[df['city'] == 'Moscow']
filtered_df = filtered_df[filtered_df['genre'] == 'pop']
filtered_df = filtered_df[filtered_df['total_play'] > 10.0]

# хуже
filtered_df = df[df['city'] == 'Moscow']
temp = filtered_df[filtered_df['genre'] == 'pop']
one_more_temp = temp[temp['total_play'] > 10.0]
one_more_temp


Если срез начинается с первого элемента, не стоит указывать 0 в срезе:
# хорошо
first_genres = df['genres'][:10]

# хуже
first_genres = df['genres'][0:10] 