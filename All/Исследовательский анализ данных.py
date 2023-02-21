Знакомство с задачей
Как исследовать поведение водителей на заправках, если вы — Яндекс? Обратиться к статистике Яндекс.Навигатора: узнать, на какую именно заправку заезжал водитель и сколько времени там провёл.
Ваши коллеги из Навигатора собрали необходимые данные и прислали их в таком виде:
Зашифрованное наименование сети АЗС (столбец name): вместо брендов — названия растений.
Уникальный идентификатор конкретной АЗС (столбец id) — в сети их много.
Время заезда на АЗС (столбец date_time) в формате ISO: 20190405T165358 означает, что водитель прибыл на заправку 5 апреля 2019 года в 16 часов 53 минуты 58 секунд по UTC.
Проведённое на АЗС время (столбец time_spent) в секундах.
Нужно ответить на вопрос, сколько в среднем времени тратят водители на заправку в каждой из сетей АЗС.

Задача 
Прочитайте файл visits.csv из папки /datasets/, указав в качестве разделителя знак табуляции \t, и сохраните результат в датафрейме data. Выведите его первые пять строк.
Путь к файлу: /datasets/visits.csv

import pandas as pd
data = pd.read_csv('/datasets/visits.csv', sep = '\t')
print(data.head(5))

Применяем сводные таблицы

Задача 
Сводные таблицы используют на разных этапах работы с данными. Можно начать с оценки данных и посчитать среднее время заправки в секундах. Нужные значения хранит столбец time_spent.
С помощью pivot_table() вычислите среднее время, проведённое на заправках в каждой из сетей, и сохраните результат в переменную name_stat.
Выведите на экран значение переменной name_stat и проанализируйте полученные данные. Не забудьте, что time_spent хранит значения в секундах.

import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')

name_stat = data.pivot_table(index = ['name'],  values ='time_spent')
print(name_stat)

Базовая проверка данных

В работе с данными почти всегда вас ждут сюрпризы:
Почему-то выгрузили не те данные или не всё, что есть.
Ошибки в алгоритмах, считающих заезды: скажем, время заправки учли неверно.
Не тот формат: например, вместо секунд записали минуты.
Упущен какой-нибудь существенный факт. Так, водители могли заехать на нерабочую АЗС (а счётчик их учёл) и развернуться, не заправившись (счётчик зафиксировал очень короткое время).

1. Сперва найдите количество заездов на АЗС. Одна строка в датафрейме соответствует одному посещению, значит, нужно посчитать строки.
Сохраните количество строк датафрейма в переменную total_visits. Результат выведите на экран так:

import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')
total_visits = data['id'].count()
print('Количество заездов:',total_visits)

2.  Теперь нужно понять, сколько АЗС в данных. У каждой станции есть свой номер — id. Чтобы найти количество АЗС, посчитайте уникальные id.

import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')
total_visits = data['id'].count()
print('Количество заездов:',total_visits)
total_stations =  len(data['id'].unique())
print('Количество АЗС:', total_stations)


3. Аналитику могут сообщить, за какой срок собрали данные. Эту информацию лучше перепроверить. Понадобится столбец date_time, который хранит время прибытия водителей на АЗС. 
Выведите минимальное и максимальное значения столбца date_time через пробел, вызвав функцию print() только один раз. 
Добавлять к выводу дополнительный текст или сохранять значения в переменные не нужно.

import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')
total_visits = data['id'].count()
print('Количество заездов:',total_visits)
total_stations =  len(data['id'].unique())
print('Количество АЗС:', total_stations)
print(data["date_time"].min(),data["date_time"].max())


4. Записи в столбце date_time хранятся в формате ISO: YYYYMMDDTHHMMSS. T — разделитель между датой и временем. В предыдущей задаче вы обнаружили, что первая дата прибытия на АЗС — 2 апреля 2018 года в 00:00, а последняя — 8 апреля 2018 года в 23:59. Значит, данные покрывают семь дней. Теперь можно найти среднее количество посещений АЗС за день.
Сохраните в переменную total_days количество дней.
В переменную station_visits_per_day запишите среднее количество визитов на АЗС за день. Чтобы посчитать среднее, используйте значения переменных total_visits, total_stations и total_days .
import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')
total_visits = data['id'].count()
print('Количество заездов:',total_visits)
total_stations =  len(data['id'].unique())
print('Количество АЗС:', total_stations)
print(data["date_time"].min(),data["date_time"].max())
total_days = 7
station_visits_per_day = total_visits/total_stations/total_days
print('Количество заездов на АЗС в сутки:', station_visits_per_day)

5. Вы только что нашли среднее количество заездов за день. Но будьте осторожны со средними значениями. На них влияет даже небольшое количество экстремально малых или больших значений в данных. Поэтому важно смотреть на общее распределение.
Проверьте распределение числа заездов по сетям АЗС. Можно ожидать, что больше заездов будет на популярных станциях. Выведите на экран 10 сетей АЗС с наибольшим количеством заездов и отсортируйте данные по убыванию числа посещений.
Посчитайте количество уникальных значений в столбце name.
Убедитесь, что данные отсортированы в порядке убывания, и выведите первые 10 строк.
import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')
total_visits = data['id'].count()
print('Количество заездов:',total_visits)
total_stations =  len(data['id'].unique())
print('Количество АЗС:', total_stations)
print(data["date_time"].min(),data["date_time"].max())
total_days = 7
station_visits_per_day = total_visits/total_stations/total_days
print('Количество заездов на АЗС в сутки:', station_visits_per_day)
print(data['name'].value_counts().head(10))

Гистограмма
1. Медианные и средние значения недостаточно характеризуют данные. Настало время посмотреть на распределение значений.
Постройте гистограмму по значениям времени, проведённого на АЗС. Эти значения хранятся в столбце time_spent.

import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')
data['time_spent'].hist()

2. Предыдущая гистограмма выглядит странно, потому что максимальное значение столбца time_spent сильно превышает большинство других значений.
Измените код, чтобы гистограмма стала более информативной.
Постройте новую гистограмму и увеличьте число корзин до 100.

import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')
data['time_spent'].hist(bins=100)


3. Итак, гистограмма стала более информативной. Можно исключить слишком большие значения времени заправки и посмотреть на остальные.
Используйте параметр range, чтобы изучить распределение значений time_spent, находящихся в диапазоне от 0 до 1500. Количество корзин оставьте прежним — 100.

import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')
data['time_spent'].hist(range=(0, 1500), bins=100)

Гистограмма для двух кубиков

1. Будем бросать на стол 10 монет и считать количество выпавших орлов. Функции для имитации одного броска и подсчёта числа орлов в нескольких бросках уже в прекоде. 
Мы повторили эксперимент 1000 раз и сохранили результат в переменной df_experiments.
Постройте гистограмму полученных значений с диапазоном значений от 0 до 10 и количеством корзин 11.

import random
import pandas as pd

# Функция, имитирующая один бросок монеты.
# От англ. coin - монета, flip - бросок монеты.
def coin_flip():
    # возможны два варианта:
    # - выпала решка, это +0 орлов
    # - выпал орёл, это +1 орёл
    score = random.randint(0, 1)
    return score


# Функция для суммирования числа орлов в нескольких бросках.
# Орёл и решка переводятся на английский как heads и tails.
# Аргумент repeat говорит, сколько раз бросать монету
# (от англ. repeat - повторение).
def flips_heads(repeat):
    total = 0
    for i in range(repeat):
        flip = coin_flip()
        total += flip
    return total


# Cоздаём пустой список. В него мы
# будем складывать результаты экспериментов.
experiments = []

for i in range(1000):
    score = flips_heads(10)

    # Напомним: функция append() добавляет новый
    # элемент score в конец списка experiments.
    experiments.append(score)

# превращаем список в DataFrame
df_experiments = pd.DataFrame(experiments)


# постройте гистограмму для df_experiments
df_experiments.hist(bins= 11, range=(0, 10))

2. Валерик каждый день едет на работу с тремя пересадками: сперва на автобусе до метро; затем по одной ветке, а потом по другой; и от метро добирается до работы на автобусе. 
Валерик знает, сколько продолжается поездка на каждом виде транспорта и пересадки, но вечно забывает учесть ожидание автобусов и поездов. Постройте гистограмму опозданий Валерика за 5 лет c параметром bins=10.
Будем считать, что автобус прибывает за время от 0 до 10 минут, а поезд — за время от 0 до 5 минут. В прекоде мы уже написали функции, которые это имитируют.

import random
import pandas as pd

# Функция, имитирующая время ожидания автобуса.
# (от англ. bus - автобус, wait - ждать, time - время)
def bus_wait_time():
    return random.randint(0, 10)


# Функция, имитирующая время ожидания поезда в метро.
# (от англ. train - поезд, wait - ждать, time - время)
def train_wait_time():
    return random.randint(0, 5)


# Функция подсчёта полного опоздания за день.
# от англ. total - полный, итоговый, delay - задержка, опоздание
def total_delay():
    # обратите внимание, как в Python переносят длинные строки
    return (
        bus_wait_time()
        + train_wait_time()
        + train_wait_time()
        + bus_wait_time()
    )


# создайте пустой список
days = []

for i in range(365 * 5):
    delay = total_delay()
    days.append(delay)
    # добавьте опоздание в список days
    

# превращаем список в DataFrame
df_days = pd.DataFrame(days)

# постройте гистограмму для df_days
df_days.hist(range = (0,10), bins =10)


Распределения
Диаграмма размаха
Диаграмма размаха в Python

Задача 
Нарисуйте диаграмму размаха для data, ограничив диапазон по вертикали значениями -100 и 1000.
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')

plt.ylim(-100, 1000)
data.boxplot() 


Описание данных

Задача
Примените к таблице data метод describe() и выведите на экран список характерных значений.

import pandas as pd

data = pd.read_csv ('/datasets/visits.csv', sep='\t')
print(data.describe())


Срезы данных и поиск авиабилетов
1. Выберите строки с выгодной ценой за авиабилет. Выгодными считаются те билеты, которые дешевле самого дорогого билета более чем в 1,5 раза.
Выведите на экран полученную выборку.

import pandas as pd

df = pd.DataFrame(
    {
        'From': [
            'Moscow',
            'Moscow',
            'St. Petersburg',
            'St. Petersburg',
            'St. Petersburg',
        ],
        'To': ['Rome', 'Rome', 'Rome', 'Barcelona', 'Barcelona'],
        'Is_Direct': [False, True, False, False, True],
        'Has_luggage': [True, False, False, True, False],
        'Price': [21032, 19250, 19301, 20168, 31425],
        'Date_From': [
            '01.07.19',
            '01.07.19',
            '04.07.2019',
            '03.07.2019',
            '05.07.2019',
        ],
        'Date_To': [
            '07.07.19',
            '07.07.19',
            '10.07.2019',
            '09.07.2019',
            '11.07.2019',
        ],
        'Airline': ['Belavia', 'S7', 'Finnair', 'Swiss', 'Rossiya'],
        'Travel_time_from': [995, 230, 605, 365, 255],
        'Travel_time_to': [350, 225, 720, 355, 250],
    }
)
print(df[df['Price'] * 1.5 < df['Price'].max()]) # впишите нужное условие

2. Выберите строки, где значения столбца Travel_time_from больше или равны 365 или значения Travel_time_to меньше 250.

import pandas as pd

df = pd.DataFrame(
    {
        'From': [
            'Moscow',
            'Moscow',
            'St. Petersburg',
            'St. Petersburg',
            'St. Petersburg',
        ],
        'To': ['Rome', 'Rome', 'Rome', 'Barcelona', 'Barcelona'],
        'Is_Direct': [False, True, False, False, True],
        'Has_luggage': [True, False, False, True, False],
        'Price': [21032, 19250, 19301, 20168, 31425],
        'Date_From': [
            '01.07.19',
            '01.07.19',
            '04.07.2019',
            '03.07.2019',
            '05.07.2019',
        ],
        'Date_To': [
            '07.07.19',
            '07.07.19',
            '10.07.2019',
            '09.07.2019',
            '11.07.2019',
        ],
        'Airline': ['Belavia', 'S7', 'Finnair', 'Swiss', 'Rossiya'],
        'Travel_time_from': [995, 230, 605, 365, 255],
        'Travel_time_to': [350, 225, 720, 355, 250],
    }
)
print(df[(df['Travel_time_from'] >= 365 ) | (df['Travel_time_to'] < 250)])

3. Выберите строки, где:
    полёт с пересадкой;
    прилёт до 8 июля (ни 9, ни 10, ни 11 июля).


import pandas as pd

df = pd.DataFrame(
    {
        'From': [
            'Moscow',
            'Moscow',
            'St. Petersburg',
            'St. Petersburg',
            'St. Petersburg',
        ],
        'To': ['Rome', 'Rome', 'Rome', 'Barcelona', 'Barcelona'],
        'Is_Direct': [False, True, False, False, True],
        'Has_luggage': [True, False, False, True, False],
        'Price': [21032, 19250, 19301, 20168, 31425],
        'Date_From': [
            '01.07.19',
            '01.07.19',
            '04.07.2019',
            '03.07.2019',
            '05.07.2019',
        ],
        'Date_To': [
            '07.07.19',
            '07.07.19',
            '10.07.2019',
            '09.07.2019',
            '11.07.2019',
        ],
        'Airline': ['Belavia', 'S7', 'Finnair', 'Swiss', 'Rossiya'],
        'Travel_time_from': [995, 230, 605, 365, 255],
        'Travel_time_to': [350, 225, 720, 355, 250],
    }
)

print(df[~(df['Is_Direct']) & ~(df['Date_To'].isin(('09.07.2019', '10.07.2019', '11.07.2019')))]) # впишите нужное условие



Срезы данных методом query()
Возможности query()
1. Выберите строки, где: Has_luggage равно False и Airline не равно ни S7, ни Rossiya. 
Напечатайте полученную выборку на экране.
import pandas as pd

df = pd.DataFrame(
    {
        'From': [
            'Moscow',
            'Moscow',
            'St. Petersburg',
            'St. Petersburg',
            'St. Petersburg',
        ],
        'To': ['Rome', 'Rome', 'Rome', 'Barcelona', 'Barcelona'],
        'Is_Direct': [False, True, False, False, True],
        'Has_luggage': [True, False, False, True, False],
        'Price': [21032, 19250, 19301, 20168, 31425],
        'Date_From': [
            '01.07.19',
            '01.07.19',
            '04.07.2019',
            '03.07.2019',
            '05.07.2019',
        ],
        'Date_To': [
            '07.07.19',
            '07.07.19',
            '10.07.2019',
            '09.07.2019',
            '11.07.2019',
        ],
        'Airline': ['Belavia', 'S7', 'Finnair', 'Swiss', 'Rossiya'],
        'Travel_time_from': [995, 230, 605, 365, 255],
        'Travel_time_to': [350, 225, 720, 355, 250],
    }
)
print(df.query('~Has_luggage  and (Airline not in ["S7","Rossiya"])')) # впишите условие создания нужной выборки

2. Выберите строки, где Airline равно Belavia, S7 или Rossiya, при этом Travel_time_from меньше переменной под названием max_time. 
Напечатайте полученную выборку на экране.

import pandas as pd

df = pd.DataFrame(
    {
        'From': [
            'Moscow',
            'Moscow',
            'St. Petersburg',
            'St. Petersburg',
            'St. Petersburg',
        ],
        'To': ['Rome', 'Rome', 'Rome', 'Barcelona', 'Barcelona'],
        'Is_Direct': [False, True, False, False, True],
        'Has_luggage': [True, False, False, True, False],
        'Price': [21032, 19250, 19301, 20168, 31425],
        'Date_From': [
            '01.07.19',
            '01.07.19',
            '04.07.2019',
            '03.07.2019',
            '05.07.2019',
        ],
        'Date_To': [
            '07.07.19',
            '07.07.19',
            '10.07.2019',
            '09.07.2019',
            '11.07.2019',
        ],
        'Airline': ['Belavia', 'S7', 'Finnair', 'Swiss', 'Rossiya'],
        'Travel_time_from': [995, 230, 605, 365, 255],
        'Travel_time_to': [350, 225, 720, 355, 250],
    }
)
max_time = 300
print(df.query('Airline in ["Belavia","S7","Rossiya"] and Travel_time_from < @max_time')) # впишите условие создания нужной выборки


Срезы в действии

1. Итак, нужно разобраться с аномалиями в выборке. Для начала найдите АЗС с самыми большими значениями в столбце time_spent.
Одной строкой кода упорядочьте столбец time_spent по убыванию и выведите на экран первые 10 строк всей таблицы.
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/datasets/visits.csv', sep='\t') 

print(data.sort_values(by='time_spent', ascending=False).head(10))

2. Четыре из десяти самых долгих заездов произошли на станции под номером 3c1e4c52. Аналитик данных непременно спросит: «А как распределение времени, проведённого на этой АЗС, соотносится с распределением времени заездов в целом?» Нужно проверить. Для этого сделайте срез данных и извлеките всю информацию о станции 3c1e4c52.
Сделайте срез data по АЗС с id == "3c1e4c52" и сохраните результат в переменную sample.
Выведите на экран число заездов на эту АЗС.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
sample = data.query('id=="3c1e4c52"')
print(len(sample))


3. Нужно сравнить распределение времени пребывания на станции 3c1e4c52 с распределением времени пребывания на всех АЗС. 
Если они сильно различаются, возможно, станция 3c1e4c52 представляет собой статистический выброс.
Методом hist() постройте две гистограммы распределения значений в столбце time_spent: одну для объекта data, вторую — для sample. 
Не забудьте использовать plt.show() после каждого вызова hist().
Для обеих гистограмм задайте одинаковые аргументы: range — от 0 до 1500, bins — 100.

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/datasets/visits.csv', sep='\t')
sample = data.query('id == "3c1e4c52"')


data['time_spent'].hist(bins=100, range=(0,1500))
plt.show()
sample['time_spent'].hist(bins=100, range=(0,1500))
plt.show() 


Работа с датой и временем

import pandas as pd 

df = pd.DataFrame({'time': ['2011-03-01 17:34']})
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M')
df['time_rounded'] = df['time'].dt.round('1H') # округляем до ближайшего значения с шагом в один час
print(df['time_rounded'])

1.Причиной коротких заездов может быть то, что водители нечаянно заезжают на АЗС, которые не работают по ночам. 
Если это действительно так, то вы увидите связь между короткими заездами и временем прибытия. 
Чтобы проверить эту гипотезу, измените тип столбца date_time на более удобный тип для даты.
Методом pd.to_datetime() переведите значения столбца date_time в таблице data в объекты datetime. 
В параметре format= укажите строку, соответствующую текущему формату date_time, с помощью специальных обозначений.
Выведите на экран первые пять строк data, чтобы проверить, что получилось.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y%m%dT%H%M%S')
print(data.head())


2. Напомним, что в датафрейме записано время UTC. Московское рассчитывают как UTC + 3 часа.
Создайте столбец data['local_time'] и сохраните в нём сдвинутое на 3 часа время из столбца data['date_time'].
Напечатайте первые 5 строк таблицы data.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y%m%dT%H%M%S')
data['local_time'] = data['date_time'] + pd.Timedelta(hours=3)
print(data.head())

3. Данные, связанные со временем, лучше округлять до той величины, которой будет достаточно для детального анализа. 
Чтобы проанализировать взаимосвязь между временем прибытия на АЗС и продолжительностью заезда, точность до минут и секунд не нужна.
Округлите время до часов.
Выполните следующие шаги:
    Создайте новый столбец date_hour и передайте ему значения столбца local_time, округлённые до часов.
    Выведите первые пять строк data, чтобы проверить результаты.
   
import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y%m%dT%H%M%S')
data['local_time'] = data['date_time'] + pd.Timedelta(hours=3)
data['date_hour'] = data['local_time'].dt.round('1H') 
print(data.head())


Графики

Задача
Снова создайте переменную sample, записав в неё срез из данных по АЗС с id == '3c1e4c52'.
Обратите внимание, что на этот раз в sample войдут все форматы времени.
Пользуясь данными sample, постройте график зависимости продолжительности заправки от времени заезда. 
За основу возьмите соответствующие столбцы time_spent и local_time. Оси X присвойте значения столбца local_time, а оси Y — значения столбца time_spent.
Проверьте, всё ли верно отображено на графике:
    Каждый элемент обозначен точкой.
    Диапазон оси Y указан от 0 до 1000.
    Добавлены сетки.
    Размер графика 12х6 дюймов.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
data['date_time'] = pd.to_datetime(
    data['date_time'], format='%Y-%m-%dT%H:%M:%S'
)
data['local_time'] = data['date_time'] + pd.Timedelta(hours=3)
sample = data.query('id == "3c1e4c52"')
sample = data.query('id == "3c1e4c52"')
sample.plot(
    x='local_time',
    y='time_spent',
    ylim=(0, 1000),
    style='o',
    grid=True,
    figsize=(12, 6),
) 


Группировка с pivot_table()

Задача
Если между временем прибытия на АЗС и числом заездов нет никакой связи, это серьёзный повод насторожиться.
Вряд ли количество заездов в два часа ночи и в восемь утра одинаково.
Чтобы понять, что же происходит, постройте график зависимости между временем прибытия и количеством заездов в час.
Выполните следующие шаги, помня о бритве Оккама:
Сделайте срез из data по АЗС с id=="3c1e4c52".
Из данных этого среза постройте сводную таблицу, которая будет отображать количество заездов по времени прибытия.
Из данных этой таблицы постройте график зависимости между временем прибытия (ось X) и количеством заездов в час (ось Y). Добавьте линии сетки, задайте размер графика 12х5 дюймов.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
data['local_time'] = pd.to_datetime(
    data['date_time'], format='%Y-%m-%dT%H:%M:%S'
) + pd.Timedelta(hours=3)
data['date_hour'] = data['local_time'].dt.round('1H')

(
    data.query('id=="3c1e4c52"')
    .pivot_table(index='date_hour', values='time_spent',aggfunc='count')
    .plot(grid=True, figsize=(12, 5))
)

Помечаем срез данных

1. Первым делом нужно создать переменную, чтобы выделить аномально быстрые заезды.
Добавьте в таблицу data столбец too_fast (пер. «слишком быстрый») со значениями:
    True — если продолжительность заезда из столбца time_spent менее 60 секунд.
    False — все остальные значения.
Затем выведите на экран первые пять строк таблицы data, чтобы проверить новый столбец.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
data['too_fast']=data['time_spent'] < 60
print(data.head())


2. Рассчитать процент всех заездов короче 60 секунд можно разными способами.
Можно посчитать значения True в столбце too_fast методом value_counts() и разделить получившееся число на количество строк.
Другой способ — применить к столбцу too_fast метод mean(). Ведь среднее рассчитывают так: сумму значений делят на количество значений. Если применить арифметическую операцию к булевым значениям True и False, значение True будет интерпретировано как 1, а False — как 0. С помощью mean() можно сделать оба вычисления сразу: посчитать True и разделить его на количество строк.
Таким образом, найти процент быстрых заездов можно с помощью среднего арифметического.
Рассчитайте среднее арифметическое для значений в столбце too_fast и выведите его на экран.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
data['too_fast'] = data['time_spent'] < 60
print(data['too_fast'].mean())

3. Переменная задана, процент посчитан, теперь можно группировать данные по АЗС.
Для этого воспользуйтесь сводной таблицей.
Создайте переменную too_fast_stat и запишите в неё значения из сводной таблицы, сгруппировав доли быстрых заездов по АЗС.
Выведите на экран первые пять строк too_fast_stat.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
data['too_fast'] = data['time_spent'] < 60
too_fast_stat = data.pivot_table(index = 'id', values ='too_fast', aggfunc='mean')
print(too_fast_stat.head())



4. Теперь вы знаете, сколько быстрых заездов на первых пяти АЗС в процентном отношении.
Но что делать дальше — выводить на экран остальные 466 строк и изучать значения для каждой АЗС? Слишком сложно.
Гораздо лучше визуализировать распределение быстрых заездов сразу по всем АЗС. Гистограмма, вот что нужно!
Постройте гистограмму распределения значений в таблице too_fast_stat на 30 корзин.
import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
data['too_fast'] = data['time_spent'] < 60
too_fast_stat = data.pivot_table(index='id', values='too_fast')
too_fast_stat.hist(bins=30)

5. Теперь, когда вы разобрались, как использовать булевы значения для подсчёта процентов, примените этот метод для аномально долгих заправок — проверьте их распределение по АЗС. Как вы помните, заезды длиннее 1000 секунд решили исключить. Сейчас станет понятно, сколько АЗС это затронет.
Добавьте в data столбец too_slow (пер. «слишком медленный»), в котором значения из столбца time_spent больше 1000 секунд будут отмечены как True, а все остальные — как False.
Помня о бритве Оккама:
Создайте сводную таблицу с процентом медленных заездов для каждой АЗС.
Постройте гистограмму доли медленных заездов по всем АЗС на 30 корзин.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')
data['too_fast'] = data['time_spent'] < 60
too_fast_stat = data.pivot_table(index='id', values='too_fast')
too_fast_stat.hist(bins=30)
data['too_slow'] = data['time_spent'] > 1000
too_slow_stat = data.pivot_table(index='id', values='too_slow')
too_slow_stat.hist(bins=30)

Срез по данным из внешнего словаря
1. Проблема: в выборке есть АЗС, на которых длительность большинства заездов короче 60 секунд.
Что нужно сделать: исключить эти АЗС из анализа.
Так будет меньше шансов получить необъективные результаты, поскольку коротких заездов в целом больше всего.
Правило: исключаются из анализа те АЗС, на которых длительность половины или более заездов короче 60 секунд.
Если бы вы писали отчёт, это предложение непременно вошло бы в него.
Примените правило на практике. 
Сначала из таблицы too_fast_stat получите id станций, которые не нарушают правило. Затем используйте эти id, чтобы отфильтровать таблицу data.
Выполните следующие шаги:
    Создайте переменную good_ids и поместите в неё те строки из too_fast_stat, где too_fast меньше 50%. 
    Не забывайте, что в таблице data too_fast является булевым значением и атрибутом заезда, но в таблице too_fast_stat too_fast — это процент заездов, длительность которых короче 60 секунд, и атрибут заправочной станции.
    
    Создайте переменную good_data и поместите в неё те строки из data, где в good_ids.index находится id. 
    Другими словами, соберите все заезды, не нарушающие правило.
    Распечатайте число строк в data, а затем и число строк в good_data.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')

good_ids  = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
print(data['id'].count())
print(good_data['id'].count())

2. Проблема: в выборке есть аномально быстрые и аномально медленные заезды. 
Скорее всего, их совершали не те, кто приезжал просто заправиться.
Что нужно сделать: исключить эти заезды из анализа, чтобы получить более точные показатели. Аномальные значения могут влиять на средние значения и медианы.
Правило: заезды, длительность которых короче 60 секунд и длиннее 1000 секунд, исключаются из анализа — это предложение тоже вошло бы в отчёт.
Примените новое правило и получите выборку без аномальных заездов.
Выполните следующие шаги:
    С помощью функции query() обновите таблицу good_data, выбрав строки, где time_spent в диапазоне между 60 и 1000 секунд. 
    Убедитесь, что заезды длительностью по 60 и 1000 секунд тоже включены.
    Выведите на экран число строк в обновлённой good_data.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')

good_ids  = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('1000 >= time_spent >= 60')
print(good_data['id'].count())


3. В предыдущих задачах вы выбросили из набора данных отдельные заезды и даже целые АЗС ради более реалистичной оценки. 
Проверьте, помогло ли это. Постройте гистограмму распределения медианной длительности заправки по всем АЗС.
Выполните следующие шаги:
Создайте переменную good_stations_stat и поместите в неё данные из таблицы с медианными значениями time_spent по каждой АЗС.
Не забудьте рассчитать медианы по good_data.
Постройте гистограмму на 50 корзин по этим медианным значениям.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index and 60 <= time_spent <= 1000')

# напишите код для расчёта медиан и построения гистограммы на 50 корзин
good_stations_stat  =  good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat.hist(bins=50)

4. Как вы помните, цель вашего анализа — определить сети заправок, где водители задерживаются надолго. 
Рассчитайте медианную продолжительность заезда для каждой сети и выведите на экран список в порядке возрастания.
Выполните следующие шаги:
    Создайте переменную good_stat и поместите в неё данные из таблицы с медианными значениями time_spent в каждой сети, то есть по соответствующим названиям name.
    Рассчитайте медианы по good_data.
    Выведите на экран таблицу good_stat, отсортировав её по возрастанию медианного времени.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index and 60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stat =  good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
print(good_stat.sort_values(by ='time_spent'))


#Добавляем столбец (продолжение)
Чтобы быть уверенными в данных, пришлось удалить почти 52% заездов на АЗС. 
Теперь посмотрите, как «типичные» средняя и медианная длительности заправки различаются в зависимости от данных: сырых или отфильтрованных. 
Для этого выведите на экран одну таблицу, в которой для каждой сети АЗС будут показаны 
средняя длительность заправки из сырых данных из таблицы stat и медианная длительность заправки из отфильтрованных данных из таблицы good_stat.
Выполните следующие шаги:
    Создайте в таблице stat новый столбец good_time_spent с медианной длительностью заправки, рассчитанной по отфильтрованным данным из таблицы good_stat.
    Выведите на экран таблицу stat и сравните показатели.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index and 60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')

stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent']=good_stat['time_spent']
print(stat)


Объединяем данные из двух таблиц
Пора посмотреть, как заезды распределяются внутри сетей. Для этого про каждую АЗС нужно знать следующее: к какой сети она относится и сколько раз в общей сложности на неё заезжали. 
Для начала создайте таблицу с этой информацией.
Выполните следующие шаги:
Создайте переменную id_name, которая для каждой АЗС хранит информацию о названии сети и общем числе заездов. Используйте good_data , чтобы создать эту таблицу.
Выведите на экран первые пять строк id_name.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index and 60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')

stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']
id_name =  good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
print(id_name.head())


Переименование столбцов

Облегчите себе анализ и жизнь, избавившись от двухэтажных названий столбцов. Сделайте их одноэтажными и переименуйте.
Выполните следующие шаги:
Измените названия столбцов в id_name на name и count.
Выведите на экран первые пять строк датафрейма, чтобы проверить результат.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index and 60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')

stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count'] 
print(id_name.head())


Объединение столбцов методами merge() и join()

1.Предупреждаем, сейчас будет запутанно. В предыдущих уроках вы рассчитывали медианы по АЗС. Теперь нужно рассчитать медиану этих медиан по каждой сети. 
Это даст ещё один показатель «типичной» медианной длительности заездов в каждой сети: медиану распределения медианной длительности заездов на АЗС.
Из этого распределения медиан нужно будет исключить медианные значения, рассчитанные для АЗС с совсем небольшим числом заездов. 
Создайте таблицу со статистикой по АЗС, с помощью которой выявите и отфильтруйте эти лишние станции.
Выполните следующие шаги:
Создайте переменную station_stat_full, которая для каждой АЗС хранит название сети, число заездов и лучший показатель медианной длительности заправки. 
Подсказка: название сети и число заездов есть в id_name, а лучший показатель медианной длительности заправки — в good_stations_stat. Объедините эти две таблицы.
Выведите на экран первые 5 строк, чтобы посмотреть новую таблицу.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')

stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']

station_stat_full =  id_name.merge(good_stations_stat, on ='id')
print(station_stat_full.head())


2. В статистике такое часто бывает: суммарные значения, полученные из малого количества данных, оказываются ненадёжными. 
Представьте, что вы вернулись к сырым данным и рассчитали медианное значение длительности для десяти случайно выбранных заездов. 
А потом повторили эту процедуру двадцать раз. 
Разброс этих двадцати медианных значений практически гарантированно будет гораздо больше, чем в том случае, если бы вы каждый раз случайным образом выбирали по сто заездов.
Медианные значения, относящиеся к небольшому числу заездов, тоже могут быть ненадёжными — их лучше удалить. Но для начала посмотрите, как число заездов распределяется по АЗС.
Выполните следующие шаги:
Используя данные из station_stat_full, постройте гистограмму числа заездов на 30 корзин.
Постройте вторую гистограмму по тем же данным, но теперь задайте диапазон от 0 до 300 заездов.
Сравните полученные гистограммы.
import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')

stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)
station_stat_full['count'].hist(bins=30)
station_stat_full['count'].hist(bins=30,range=(0, 300))


3.Изучив построенные гистограммы, вы решили исключить те АЗС, на которые в течение семи дней заезжали 30 или менее раз. 
Для этого нужно найти в таблице station_stat_full АЗС с числом заездов больше 30, сгруппировать их по названию сети и рассчитать медиану медианных значений. 
Как вы помните, медианные значения в таблице station_stat_full — это медианная длительность заправок по АЗС. Чтобы получить значение по каждой сети, возьмите медиану этих медиан.
Выполните следующие шаги:
Не прибегая к вспомогательной переменной, сделайте срез данных из таблицы station_stat_full — так вы найдёте все строки, где число заездов больше 30. 
Для каждой сети рассчитайте медиану медианного времени заезда на АЗС, а также число АЗС, из которых складывается эта новая медиана. Сохраните результат в переменной good_stat2.
Измените названия столбцов в таблице good_stat2 на median_time и stations.
Выведите на экран первые пять строк good_stat2.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')

stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)
good_stat2 = station_stat_full.query('count > 30').pivot_table(index='name', values='time_spent', aggfunc=['median','count'])
good_stat2.columns = ['median_time', 'stations']
print(good_stat2.head())

4. Снова вызовите таблицу stat, которую вы создали несколько уроков назад. 
Вы использовали её, чтобы посмотреть, насколько различаются два показателя «типичной» длительности заправки: среднее время заправки, полученное из сырых данных, 
и медианное время заправки, полученное из отфильтрованных данных. Теперь у вас есть третий показатель. Внесите его в эту таблицу и посмотрите на результаты.
Выполните следующие шаги:
Добавьте good_stat2 в stat и сохраните получившуюся таблицу под именем final_stat.
Выведите на экран final_stat полностью и сравните показатели.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')

stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']


final_stat = stat.join(good_stat2)
print(final_stat)

Взаимосвязь данных 
# номер заказа
#RUS5763643

Диаграмма рассеяния
Задача
Постройте график по данным из station_stat_full, где для каждой АЗС будет отдельная точка: 
по горизонтальной оси — число заездов на АЗС, по вертикальной — медианное время заправки. Добавьте линии сетки.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)


station_stat_full.plot(x='count',y='time_spent', kind='scatter',grid=True)


Корреляция

Задача.
По данным из таблицы station_stat_full посчитайте коэффициент корреляции Пирсона между числом заездов на АЗС и временем заправки. 
Коэффициент выведите на экран.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

print(station_stat_full['count'].corr(station_stat_full['time_spent']))


Матрица диаграмм рассеяния
Задача
Создайте переменную station_stat_multi, где для каждой АЗС будет 3 числа:
    1) среднее (не медиана) продолжительности заезда на АЗС;
    2) средняя доля быстрых заездов;
    3) средняя доля медленных заездов.
Распечатайте матрицу корреляции между этими величинами. 
Постройте диаграмму рассеяния попарно для всех величин методом scatter_matrix(). Задайте размер 9х9 дюймов.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

station_stat_multi = data.pivot_table(index='id', values=['time_spent', 'too_fast', 'too_slow'], aggfunc=['mean'])
station_stat_multi.columns=['time_spent', 'too_fast', 'too_slow']
print(station_stat_multi.corr())
pd.plotting.scatter_matrix(station_stat_multi, figsize=(9, 9)) 

Как выжать максимум из очевидности
Задача
Добавьте в таблицу station_stat_multi столбец good_time_spent из данных good_stations_stat.
Распечатайте матрицу корреляции для station_stat_multi.
Постройте диаграммы рассеяния попарно для всех величин методом scatter_matrix. Задайте размер 9х9 дюймов.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

station_stat_multi = data.pivot_table(index='id', values=['time_spent', 'too_fast', 'too_slow'])

station_stat_multi['good_time_spent'] = good_stations_stat['time_spent']
print(station_stat_multi.corr())
pd.plotting.scatter_matrix(station_stat_multi, figsize=(9, 9)) 

Укрупняем группы
1.
Итак, вы выявили аномалии, отфильтровали данные, создали показатели для типичного времени заезда и изучили влияние большого числа аномальных заездов на эти показатели. 
Проверьте всё ещё разок, прежде чем делиться результатами. Для начала визуализируйте распределение лучших показателей заездов с типичной продолжительностью по сетям АЗС.
Выполните следующие шаги, используя чистый код. Не вводите вспомогательные переменные — помните о бритве Оккама:
Упорядочьте таблицу final_stat по возрастанию лучших показателей из столбца median_time. median_time— это медиана для распределения медианной продолжительности заправки по АЗС в каждой сети.
Постройте столбчатый график по значениям median_time. Задайте размер графика 10х5 дюймов.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']
final_stat = stat.join(good_stat2)
final_stat = final_stat.sort_values(by='median_time')
final_stat.plot(y='median_time',kind='bar',figsize=(10, 5))


2.Предыдущий столбчатый график отображал шесть сетей АЗС без данных: 
    «Годецию», «Лобулярию», «Нарцисс», «Обриету» и «Фасоль». Это значения NaN в final_stat, которые появились, потому что вы исключили непопулярные АЗС.
Таблица final_stat была создана объединением таблиц stat (включает все АЗС) и good_stat2 (исключает АЗС с малым числом заездов). 
Так как в join() по умолчанию левое соединение, индексы из final_stat будут идентичны индексам из stat. Поэтому любой индекс из таблицы stat, которого нет в таблице good_stat2, после объединения получит значение NaN. Наведите порядок в графике, удалив значения NaN.
Выполните следующие шаги, помня о бритве Оккама:
Отбросьте значения NaN в столбце median_time таблицы final_stat.
Упорядочьте таблицу final_stat по возрастанию значений в столбце median_time.
Постройте столбчатый график median_time. Задайте размер графика 10х5 дюймов. Добавьте линии сетки.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']
final_stat = stat.join(good_stat2)
#Отбросьте значения NaN в столбце median_time таблицы final_stat.
final_stat = final_stat.dropna(subset=['median_time']) 
#Упорядочьте таблицу final_stat по возрастанию значений в столбце median_time.
final_stat = final_stat.sort_values(by='median_time')
final_stat.plot(
    y='median_time',
    kind='bar',
    figsize=(10, 5),
    grid=True
)
print(final_stat)

3. До этого момента вы фильтровали данные по количеству заездов на одну АЗС и по длительности заправки. 
Но стоит учесть ещё одну переменную: число АЗС внутри сетей. С точки зрения маркетинга интересны и сети с большей продолжительностью заправки, и сети, в которых много АЗС.
Значит, нужно исключить те сети, в которых заправочных станций мало. А для начала посмотрите, как число заправочных станций распределяется по сетям.
Используя данные из таблицы final_stat, постройте гистограмму, отображающую число АЗС внутри сетей. Поделите значения на 100 корзин.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']
final_stat = stat.join(good_stat2)
final_stat['stations'].hist(bins=100)

4. Так как с точки зрения маркетинга небольшие сети неважны, создайте новую переменную с данными только крупных сетей.
Выполните следующие шаги:
Создайте переменную big_nets_stat и поместите в неё строки из таблицы final_stat, в которых значение переменной stations больше 10.
Выведите новую переменную на экран и изучите результат.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']
final_stat = stat.join(good_stat2)


big_nets_stat  = final_stat.query('stations  > 10')
print(big_nets_stat)

5.Теперь можно разделить все сети на две группы: «Большая восьмёрка» и «Другие». Вторая группа будет восприниматься как одна большая сеть.
Лучшие показатели средней продолжительности заправки содержатся в таблице good_stat2 и рассчитываются по данным station_stat_full (просмотрите код, чтобы вспомнить эти вычисления). 
Повторите вычисления, но вместо того, чтобы группировать данные по столбцу name, сгруппируйте данные по новому столбцу, содержащему категорию Другие.
Чтобы создать этот столбец в таблице station_stat_full, примените метод where() для сравнения столбца name в station_stat_full с индексами big_nets_stat.
Выполните следующие шаги:
Добавьте в таблицу station_stat_full новый столбец group_name.
Поместите в столбец group_name значения столбца name, если сеть присутствует в big_nets_stat. Если столбец name отсутствует, поместите в group_name значения из Другие.
Выведите на экран первые пять строк таблицы station_stat_full.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']
final_stat = stat.join(good_stat2)

big_nets_stat = final_stat.query('stations > 10')

station_stat_full['group_name'] = station_stat_full['name'].where(station_stat_full['name'].isin(big_nets_stat.index), 'Другие')
print(station_stat_full.head())

6. У вас есть категория Другие с небольшими сетями. Теперь повторите анализ, в процессе которого создали good_stat2, но в этот раз сгруппируйте данные по group_name.
Выполните следующие шаги:
    Создайте переменную stat_grouped, которая повторяет вычисления good_stat2, но группирует по group_name.
    Переименуйте столбцы в stat_grouped на time_spent и count.
    Упорядочьте stat_grouped по возрастанию значений столбца time_spent. Убедитесь, что изменение постоянное, а не временное.
    Выведите на экран stat_grouped.

    import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']
final_stat = stat.join(good_stat2)

big_nets_stat = final_stat.query('stations > 10')
station_stat_full['group_name'] = (
    station_stat_full['name']
    .where(station_stat_full['name'].isin(big_nets_stat.index), 'Другие')
)
stat_grouped = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='group_name', values='time_spent', aggfunc=['median', 'count'])
)
stat_grouped.columns = ['time_spent', 'count']
stat_grouped = stat_grouped.sort_values(by='time_spent')
print(stat_grouped)

7.
Теперь у вас есть таблица с лучшими показателями типичной продолжительности заезда для крупных сетей АЗС. 
Дальше уже команде маркетинга решать, сколько сил тратить на то, чтобы завоевать «Розу» (типичная продолжительность заезда 350 секунд, 18 заправочных станций) или 
«Василька» (типичная продолжительность заезда 252 секунды, 103 заправочные станции). 
В следующем уроке вы сделаете финальную проверку, а пока визуализируйте относительную величину этих сетей с точки зрения количества заправочных станций.
По данным stat_grouped постройте круговую диаграмму с числом АЗС в каждой сети. Задайте её размер 8x8 дюймов.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']
final_stat = stat.join(good_stat2)

big_nets_stat = final_stat.query('stations > 10')
station_stat_full['group_name'] = (
    station_stat_full['name']
    .where(station_stat_full['name'].isin(big_nets_stat.index), 'Другие')
)

stat_grouped = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='group_name', values='time_spent', aggfunc=['median', 'count'])
)
stat_grouped.columns = ['time_spent', 'count']


stat_groupe.plot(
    y = 'count',
    kind='pie',
    figsize=(8, 8)  
) 

1. Напоследок посмотрите, как продолжительность заездов распределяется по девяти сетям («Большая восьмёрка» и «Другие»). 
Загвоздка может быть вот в чём: если продолжительность сильно различается, то сравнивать показатели разных сетей будет неправильно. 
Например, если в сети больше заправок продолжительностью по 60–70 секунд, чем в других, это может понижать медианное значение.
Чтобы проверить, не происходит ли такое, сгруппируйте данные из good_data по group_name и постройте гистограммы. 
Первым делом создайте столбец для группировки.
Выполните следующие шаги:
Создайте столбец group_name в таблице good_data так же, как делали раньше в station_stat_full.
Выведите на экран первые 5 строк good_data, чтобы проверить работу нового столбца.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']
final_stat = stat.join(good_stat2)

big_nets_stat = final_stat.query('stations > 10')
station_stat_full['group_name'] = (
    station_stat_full['name']
    .where(station_stat_full['name'].isin(big_nets_stat.index), 'Другие')
)

stat_grouped = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='group_name', values='time_spent', aggfunc=['median', 'count'])
)
stat_grouped.columns = ['time_spent', 'count']

good_data['group_name'] = (
    good_data['name']
    .where(good_data['name'].isin(big_nets_stat.index), 'Другие')
)
print(good_data.head())


2.Теперь, когда есть столбец group_name, сгруппируйте good_data и постройте гистограмму, чтобы увидеть, 
как распределяется продолжительность заездов в каждой сети.
Выполните следующие шаги:
Сгруппируйте good_data по group_name, используя цикл for. Используйте в цикле переменные name и group_data.
В каждой итерации group_data вызывайте метод hist(), чтобы построить гистограмму по значениям time_spent на 50 корзин.
import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']
final_stat = stat.join(good_stat2)

big_nets_stat = final_stat.query('stations > 10')
station_stat_full['group_name'] = (
    station_stat_full['name']
    .where(station_stat_full['name'].isin(big_nets_stat.index), 'Другие')
)

stat_grouped = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='group_name', values='time_spent', aggfunc=['median', 'count'])
)
stat_grouped.columns = ['time_spent', 'count']

good_data['group_name'] = (
    good_data['name']
    .where(good_data['name'].isin(big_nets_stat.index), 'Другие')
)

for name,group_data in good_data.groupby('group_name'):
    group_data.hist('time_spent', bins=50)


3. Построенные гистограммы будут полезными, только если вы знаете, к какой именно сети относятся данные. Повторите предыдущее задание, но теперь сделайте название сети заголовком гистограмм. Как вы помните, метод plot() даёт больше вариантов для форматирования, чем метод hist().
Выполните следующие шаги:
Напишите ещё один цикл for, как в предыдущем задании. Используйте в качестве переменных цикла name и group_data.
В каждой итерации group_data вызовите метод plot(), чтобы построить гистограмму по значениям time_spent.
Поместите в название каждой гистограммы переменную цикла name и разбейте гистограммы на 50 корзин.

import pandas as pd

data = pd.read_csv('/datasets/visits.csv', sep='\t')

# фильтруем слишком быстрые и медленные заезды и АЗС
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

# считаем данные по отдельным АЗС и по сетям
station_stat = data.pivot_table(index='id', values='time_spent', aggfunc='median')
good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')
stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(index='name', values='time_spent', aggfunc='median')
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(index='id', values='name', aggfunc=['first', 'count'])
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_stations_stat)

# считаем показатели сетей из показателей АЗС,
# а не усреднённые заезды на все АЗС сети
good_stat2 = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])
)
good_stat2.columns = ['median_time', 'stations']
final_stat = stat.join(good_stat2)

big_nets_stat = final_stat.query('stations > 10')
station_stat_full['group_name'] = (
    station_stat_full['name']
    .where(station_stat_full['name'].isin(big_nets_stat.index), 'Другие')
)

stat_grouped = (
    station_stat_full
    .query('count > 30')
    .pivot_table(index='group_name', values='time_spent', aggfunc=['median', 'count'])
)
stat_grouped.columns = ['time_spent', 'count']

good_data['group_name'] = (
    good_data['name']
    .where(good_data['name'].isin(big_nets_stat.index), 'Другие')
)


for name,group_data in good_data.groupby('group_name'):
    group_data.plot(
        y = 'time_spent',
        title = name,
        kind='hist', 
        bins=50)
