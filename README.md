Анализ данных и оформление результатов
Задача: обновление Яндекс.Музыки

Дубликаты, повторяющиеся строки, выявляют методом duplicated() и подсчитывают тем же sum()
print(df.duplicated().sum())   
 подсчёт пропусков в данных
print(df.isna().sum()) 
Если обе проверки не выявили проблемы в данных, значит они готовы для анализа.

1.Файл music_log_upd.csv хранит данные, которые прошли предобработку в предыдущей теме. Прочитайте данные из файла music_log_upd.csv и выведите на экран первые 15 строк.
import pandas as pd

df = pd.read_csv('music_log_upd.csv')
print(df.head(15))


2.Получите список названий колонок таблицы df через атрибут columns. Результат выведите на экран.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')
print(df.columns)

3. С помощью метода isna() посчитайте пустые значения в таблице df. Результат сохраните в переменной na_number и выведите на экран.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')
na_number = df.isna().sum()
print(na_number)

4.Посчитайте количество дубликатов в наборе данных, сохраните результат в переменной duplicated_number. Выведите её значение на экран.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')
duplicated_number = df.duplicated().sum()
print(duplicated_number)


exoplanet.groupby('discovered').count()

exo_number = exoplanet.groupby('discovered')['radius'].count()
exo_radius_sum = exoplanet.groupby('discovered')['radius'].sum()
exo_radius_mean = exo_radius_sum/exo_number


Яндекс.Музыка: группировка данных
1.
Сгруппируйте данные по user_id и выберите колонку 'genre_name' как показатель для сравнения. Результат сохраните в переменной genre_grouping.
Методом count() посчитайте количество жанров, которые слушал каждый пользователь. Результат сохраните в переменной genre_counting.
Затем выведите на экран первые 30 строк из genre_counting.

import pandas as pd
df = pd.read_csv('music_log_upd.csv')
genre_grouping = df.groupby('user_id')['genre_name']
genre_counting = df.groupby('user_id')['genre_name'].count()
print(genre_counting.head(30))

2.
Предположим, что более широкие вкусы характерны для пользователей, которые слушают больше 50 песен. Чтобы найти такого пользователя, напишите функцию user_genres.
Эта функция:
Принимает группировку, сгруппированные данные, как значение для параметра group.
Перебирает группы, входящие в эту группировку. В каждой группе два элемента — имя группы с индексом 0 и список значений с индексом 1.
Обнаружив первую группу, в которой список (элемент с индексом 1) содержит более 50 значений, функция вернёт имя группы (элемент с индексом 0).
import pandas as pd

df = pd.read_csv('music_log_upd.csv')
genre_grouping = df.groupby('user_id')['genre_name']


def user_genres(group):
    for col in group:
        if len(col[0]) > 50:# назначьте условие: если длина столбца col с индексом 1 больше 50, тогда
            user = col[0] # в переменной user сохраняется элемент col[0]
            return user

3. Вызовите функцию user_genres и передайте ей genre_grouping. Результат — user_id меломана — сохраните в переменной search_id и выведите на экран её значение.

import pandas as pd

df = pd.read_csv('music_log_upd.csv')
genre_grouping = df.groupby('user_id')['genre_name']


def user_genres(group):
    for col in group:
        if len(col[1]) > 50:
            user = col[0]
            return user
search_id = user_genres(genre_grouping)
print(search_id)


print(exoplanet.sort_values(by='radius').head(30))
print(exoplanet[exoplanet['radius'] < 1])
print(exoplanet[exoplanet['discovered'] == 2014])

exo_small_14 = exoplanet[exoplanet['radius'] < 1]
exo_small_14 = exo_small_14[exo_small_14['discovered'] == 2014]
print(exo_small_14)
print(exo_small_14.sort_values(by='radius', ascending=False))


Яндекс.Музыка: сортировка данных
1.
У похожей на Солнце звезды телескоп Kepler открыл похожую на Землю планету. А вы нашли в данных Яндекс.Музыки меломана с уникальными данными. Он за день послушал больше 50 композиций.
Получите таблицу с прослушанными им треками.
Для этого запросите из структуры данных df строки, отвечающие сразу двум условиям:
Значение в столбце 'user_id' должно быть равно значению переменной search_id.
Время прослушивания, значение в столбце 'total_play_seconds', не должно равняться 0.
Сохраните результат в переменной music_user, выводить её значение на экран не нужно.

import pandas as pd
df = pd.read_csv('music_log_upd.csv')

genre_grouping = df.groupby('user_id')['genre_name']

def user_genres(group):
    for col in group:
        if len(col[1]) > 50:
            user = col[0]
            return user

search_id = user_genres(genre_grouping)

music_user1 = df[df['total_play_seconds'] != 0]
music_user = music_user1[music_user1['user_id'] == search_id]
print(music_user)


2.Сгруппируйте данные таблицы music_user по столбцу 'genre_name' и получите сумму значений столбца 'total_play_seconds'. Сохраните результат в переменной sum_music_user и выведите её значение на экран.

import pandas as pd
df = pd.read_csv('music_log_upd.csv')

genre_grouping = df.groupby('user_id')['genre_name']

def user_genres(group):
    for col in group:
        if len(col[1]) > 50:
            user = col[0]
            return user

search_id = user_genres(genre_grouping)

music_user1 = df[df['total_play_seconds'] != 0]
music_user = music_user1[music_user1['user_id'] == search_id]
#print(music_user)

sum_music_user  = music_user.groupby('genre_name')['total_play_seconds'].sum()
print(sum_music_user)

3. Предпочтения меломана начинают проявляться. Но, возможно, длительность композиций от жанра к жанру сильно различается. Важно знать, сколько треков каждого жанра он включил.
Сгруппируйте данные по столбцу genre_name и посчитайте значения в столбце genre_name. Сохраните результат в переменной count_music_user и выведите её значение на экран.
Вывод на экран из предыдущего задания закомментируйте.

import pandas as pd
df = pd.read_csv('music_log_upd.csv')

genre_grouping = df.groupby('user_id')['genre_name']

def user_genres(group):
    for col in group:
        if len(col[1]) > 50:
            user = col[0]
            return user

search_id = user_genres(genre_grouping)

music_user1 = df[df['total_play_seconds'] != 0]
music_user = music_user1[music_user1['user_id'] == search_id]
#print(music_user)

sum_music_user  = music_user.groupby('genre_name')['total_play_seconds'].sum()
#print(sum_music_user)

count_music_user  = music_user.groupby('genre_name')['genre_name'].count()
print(count_music_user)



4.
Чтобы предпочтения были видны сразу, самые крупные значения нужно расположить сверху.
Отсортируйте данные в группировке sum_music_user по убыванию. Когда применяете метод sort_values() к Series с единственным столбцом, аргумент by указывать не нужно — только порядок сортировки.
Сохраните результат в переменной final_sum и выведите её значение на экран.
Вывод на экран из предыдущего задания закомментируйте.

import pandas as pd
df = pd.read_csv('music_log_upd.csv')

genre_grouping = df.groupby('user_id')['genre_name']

def user_genres(group):
    for col in group:
        if len(col[1]) > 50:
            user = col[0]
            return user

search_id = user_genres(genre_grouping)

music_user1 = df[df['total_play_seconds'] != 0]
music_user = music_user1[music_user1['user_id'] == search_id]
#print(music_user)

sum_music_user  = music_user.groupby('genre_name')['total_play_seconds'].sum()
#print(sum_music_user)

count_music_user  = music_user.groupby('genre_name')['genre_name'].count()
#print(count_music_user)
final_sum = sum_music_user.sort_values( ascending=False)
print(final_sum)

5.Теперь то же самое сделайте с числом прослушанных меломаном композиций. Отсортируйте данные группировки count_music_user по убыванию. Сохраните результат в переменной final_count, значение которой выведите на экран.
Вывод на экран из предыдущего задания закомментируйте.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')

genre_grouping = df.groupby('user_id')['genre_name']

def user_genres(group):
    for col in group:
        if len(col[1]) > 50:
            user = col[0]
            return user

search_id = user_genres(genre_grouping)

music_user1 = df[df['total_play_seconds'] != 0]
music_user = music_user1[music_user1['user_id'] == search_id]
#print(music_user)

sum_music_user  = music_user.groupby('genre_name')['total_play_seconds'].sum()
#print(sum_music_user)

count_music_user  = music_user.groupby('genre_name')['genre_name'].count()
#print(count_music_user)
final_sum = sum_music_user.sort_values( ascending=False)
#print(final_sum)

final_count = count_music_user.sort_values(ascending=False)
print(final_count)

Описательная статистика

print(df['total_play_seconds'].max()) 

print(df[df['total_play_seconds'] == df['total_play_seconds'].max()]) 

df_drop_null = df[df['total_play_seconds'] != 0] 
print(df_drop_null['total_play_seconds'].min()) 
print(df_drop_null[df_drop_null['total_play_seconds'] == df_drop_null['total_play_seconds'].min()])

ищем медиану
df_stat_1 = df.tail()
print(df_stat_1['total_play_seconds'].sort_values())

print(df_stat['total_play_seconds'].median())

df_drop_null = df[df['total_play_seconds'] != 0]
print(df_drop_null['total_play_seconds'].median())

print(df_drop_null['total_play_seconds'].mean())


Яндекс.Музыка: описательная статистика

1.Получите таблицу с композициями самого популярного жанра — pop. Затем исключите пропущенные треки — которые слушали 0 секунд.
Сохраните результат в переменной pop_music.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')
df1 = df[(df['genre_name']=='pop')]
pop_music = df1[df1['total_play_seconds']!=0]

2.Найдите максимальное время прослушивания песни в жанре pop. Сохраните результат в переменной pop_music_max_total_play и выведите её значение на экран.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')
df1 = df[(df['genre_name']=='pop')]
pop_music = df1[df1['total_play_seconds']!=0]

pop_music_max_total_play =pop_music['total_play_seconds'].max()
print(pop_music_max_total_play)

3.Получите из таблицы pop_music строку с максимальным временем прослушивания. Результат сохраните в переменной pop_music_max_info и выведите на экран.
Закомментируйте вывод результата предыдущей задачи.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')
df1 = df[(df['genre_name']=='pop')]
pop_music = df1[df1['total_play_seconds']!=0]
#print(pop_music_max_total_play)
pop_music_max_total_play =pop_music['total_play_seconds'].max()
pop_music_max_info  = pop_music[pop_music['total_play_seconds'] == pop_music['total_play_seconds'].max()]
print(pop_music_max_info)

4.Найдите минимальное время прослушивания композиции в жанре pop, отличное от нуля. Сохраните его в переменной pop_music_min_total_play, значение выведите на экран. 
Вывод результата предыдущей задачи закомментируйте.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')
df1 = df[(df['genre_name']=='pop')]
pop_music = df1[df1['total_play_seconds']!=0]
#print(pop_music_max_total_play)
pop_music_max_total_play =pop_music['total_play_seconds'].max()
pop_music_max_info  = pop_music[pop_music['total_play_seconds'] == pop_music['total_play_seconds'].max()]
#print(pop_music_max_info)
pop_music_min_total_play =pop_music['total_play_seconds'].min()
print(pop_music_min_total_play)

5.Выведите на экран строку о композиции жанра pop, которую начали слушать, но выключили быстрее всех остальных. Результат сохраните в переменную pop_music_min_info и выведите на экран.
Вывод результата предыдущей задачи закомментируйте.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')
df1 = df[(df['genre_name']=='pop')]
pop_music = df1[df1['total_play_seconds']!=0]
#print(pop_music_max_total_play)
pop_music_max_total_play =pop_music['total_play_seconds'].max()
pop_music_max_info  = pop_music[pop_music['total_play_seconds'] == pop_music['total_play_seconds'].max()]
#print(pop_music_max_info)
pop_music_min_total_play =pop_music['total_play_seconds'].min()
#print(pop_music_min_total_play)
pop_music_min_info   = pop_music[pop_music['total_play_seconds'] == pop_music['total_play_seconds'].min()]
print(pop_music_min_info)

6.Рассчитайте медиану времени прослушивания произведений жанра pop. Сохраните результат в переменной pop_music_median и выведите на экран.
Вывод результата предыдущей задачи закомментируйте.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')
df1 = df[(df['genre_name']=='pop')]
pop_music = df1[df1['total_play_seconds']!=0]
#print(pop_music_max_total_play)
pop_music_max_total_play =pop_music['total_play_seconds'].max()
pop_music_max_info  = pop_music[pop_music['total_play_seconds'] == pop_music['total_play_seconds'].max()]
#print(pop_music_max_info)
pop_music_min_total_play =pop_music['total_play_seconds'].min()
#print(pop_music_min_total_play)
pop_music_min_info   = pop_music[pop_music['total_play_seconds'] == pop_music['total_play_seconds'].min()]
#print(pop_music_min_info)

pop_music_median = pop_music['total_play_seconds'].median()
print(pop_music_median)

7.Рассчитайте среднее арифметическое времени прослушивания произведений жанра pop. Сохраните результат в переменной pop_music_mean и выведите на экран.
Вывод результата предыдущей задачи закомментируйте.
import pandas as pd
df = pd.read_csv('music_log_upd.csv')
df1 = df[(df['genre_name']=='pop')]
pop_music = df1[df1['total_play_seconds']!=0]
#print(pop_music_max_total_play)
pop_music_max_total_play =pop_music['total_play_seconds'].max()
pop_music_max_info  = pop_music[pop_music['total_play_seconds'] == pop_music['total_play_seconds'].max()]
#print(pop_music_max_info)
pop_music_min_total_play =pop_music['total_play_seconds'].min()
#print(pop_music_min_total_play)
pop_music_min_info   = pop_music[pop_music['total_play_seconds'] == pop_music['total_play_seconds'].min()]
#print(pop_music_min_info)
pop_music_median = pop_music['total_play_seconds'].median()
#print(pop_music_median)
pop_music_mean = pop_music['total_play_seconds'].mean()
print(pop_music_mean)

Яндекс.Музыка: решение кейса и оформление результатов

1.Рассчитайте метрику engagement после проведения эксперимента для всего набора данных. Сохраните полученный результат в переменной current_engagement и выведите на экран.
import pandas as pd
current_engagement = df.groupby('user_id').sum().median()
print(current_engagement)

2.Внесите результат своей работы в существующую таблицу и рассчитайте разность двух значений метрики engagement.
Названия столбцов:
metrics — метрика;
before_test — до эксперимента;
after_test — после эксперимента;
absolute_difference — абсолютная разница;
Значение метрики engagement после эксперимента: 62.344431 секунд.
Значение метрики engagement до эксперимента: 57.456 секунд.


import pandas as pd
exp = [['engagement', 0, 0, 0]]
exp[0][1] = 57.456
exp[0][2] = 62.344431
exp[0][3] = exp[0][2] - exp[0][1]
columns = ['metrics','before_test','after_test','absolute_difference']
metrics = pd.DataFrame(data=exp,columns=columns)
print(metrics)


3.Получите выборку прослушанных композиций в жанре рок, время воспроизведения которых не равно нулю, и сохраните её в переменной genre_rock. Получите максимальное и минимальное значения времени прослушивания, сохраните соответственно в переменных genre_rock_max и genre_rock_min, выведите на экран со строками:
'Максимальное время прослушивания в жанре рок равно:'
'Минимальное время прослушивания в жанре рок равно:'

import pandas as pd
df = pd.read_csv('music_log_upd.csv')

df1 = df[(df['genre_name']=='rock')]
genre_rock = df1[df1['total_play_seconds']!=0]
genre_rock_max  = genre_rock['total_play_seconds'].max()
genre_rock_min = genre_rock['total_play_seconds'].min()

print('Максимальное время прослушивания в жанре рок равно:', genre_rock_max)
print('Минимальное время прослушивания в жанре рок равно:', genre_rock_min)


4.Соберите результаты исследования в таблицу research_genres_result, которую нужно создать конструктором DataFrame(). 
Его аргумент data — список с данными, аргумент columns — список названий столбцов. Выведите полученную таблицу на экран.

import pandas as pd
data = [['pop', 8663, 34.6, 1158.03, 0.000794],
       ['rock', 6828, 33.3, 1699.14882, 0.014183]]
columns = ['genre_name','total_track','skip_track','max_total_time','min_total_time']

research_genres_result = pd.DataFrame(data=data,columns=columns)

print(research_genres_result)
