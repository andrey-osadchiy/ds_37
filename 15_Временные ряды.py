Временные ряды

1. Измените тип данных Datetime с object на datetime64. Но прежде запустите код и просмотрите общую информацию о данных.
В документации Pandas выберите любой способ преобразования данных. Формат вывода даты указывать не нужно: библиотека определит его самостоятельно.
Напечатайте на экране информацию о таблице (уже в прекоде).


import pandas as pd

data = pd.read_csv('/datasets/energy_consumption.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
print(data.info())


2. Установите индекс таблицы равным столбцу Datetime. В документации Pandas выберите любой способ установки индекса.
Напечатайте на экране информацию о таблице (уже в прекоде).

import pandas as pd

data = pd.read_csv('/datasets/energy_consumption.csv', parse_dates=[0],index_col = 'Datetime' )
print(data.info())

3. Чтобы проверить, в хронологическом ли порядке расположены даты и время, посмотрите атрибут индекса таблицы is_monotonic (англ. «монотонный»). Если порядок соблюдён, атрибут вернёт True, если нет — False.
Отсортируйте индекс таблицы. Метод найдите в документации.
Напечатайте на экране значение атрибута is_monotonic (уже в прекоде). Затем вызовом функции info() выведите на экран общую информацию о таблице.

import pandas as pd

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data = data.sort_index()
print(data.index.is_monotonic)
print(data.info())

4. Из временного ряда выделите данные с января по июнь 2018 года.
Даты во временных рядах можно указывать в срезах. В прекоде выбраны значения с 2016 по 2017 год включительно.
Напечатайте на экране информацию о таблице (уже в прекоде).
import pandas as pd

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
#data = data['2016':'2017']
data = data['2018-01': '2018-06' ] 
print(data.info())

5. Постройте график временного ряда.

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data['2018-01':'2018-06']
data.plot()

#Ресемплирование
1. Постройте график среднего потребления электроэнергии по годам.

import pandas as pd

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data.resample('1Y').mean()
data.plot()

2. Постройте график энергопотребления с января по июнь 2018 года. Выберите интервал в один день, по каждому — вычислите суммарное энергопотребление.

import pandas as pd

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data['2018-01':'2018-06']
data = data.resample('1D').sum()
data.plot()

#Скользящее среднее
Задача.
Добавьте в столбец 'rolling_mean' скользящее среднее с размером окна, равным 10. Выведите на экран графики энергопотребления с января по июнь 2018 года и скользящего среднего (уже в прекоде).

import pandas as pd

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data['2018-01':'2018-06'].resample('1D').sum()
data['rolling_mean'] = data.rolling(10).mean() 
data.plot()

#Тренды и сезонность
1. Разложите временной ряд на тренд и сезонную компоненту. Допишите код вывода графиков этих составляющих ряда.
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data['2018-01':'2018-06'].resample('1D').sum()

decomposed = seasonal_decompose(data) 

plt.figure(figsize=(6, 8))
plt.subplot(311)
# Чтобы график корректно отобразился, указываем его
# оси ax, равными plt.gca() (англ. get current axis,
# получить текущие оси)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca()) 
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')
plt.tight_layout()

2.Постройте график сезонной составляющей за первые 15 дней января 2018 года.

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data['2018-01':'2018-06'].resample('1D').sum()
decomposed = seasonal_decompose(data) 
plt.title('Seasonality')
decomposed.seasonal['2018-01-01':'2018-01-15'].plot(ax=plt.gca()) 

#Разности временного ряда

Вычислите разности временного ряда. Пропущенные значения заполнять не нужно.
На графике изобразите скользящее среднее и скользящее стандартное отклонение (уже в прекоде).

import pandas as pd

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data['2018-01':'2018-06'].resample('1D').sum()
data = data -  data.shift()
data['mean'] = data['PJME_MW'].rolling(15).mean()
data['std'] = data['PJME_MW'].rolling(15).std()
data.plot()

#Задача прогнозирования
Разбейте датасет о потреблении электроэнергии на обучающую и тестовую выборки в соотношении 4:1. Возьмите данные за доступное время.
Напечатайте на экране минимальные и максимальные значения индексов выборок (уже в прекоде). Они нужны, чтобы убедиться в корректности деления.

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data.resample('1D').sum()

train, test = train_test_split(data, shuffle=False, test_size=0.2)


print(train.index.min(), train.index.max())
print(test.index.min(), test.index.max())

#Качество прогноза
1. Оцените модель первым способом — прогнозом константой. Дневной объём электропотребления предскажите медианой, сохраните значения в переменной pred_median и найдите для этого прогноза значение MAE.
В прекоде указан средний объём электропотребления, чтобы вы смогли соотнести его со значением метрики MAE.
Напечатайте на экране значения среднего объёма электропотребления и метрики MAE (уже в прекоде).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data.resample('1D').sum()

train, test = train_test_split(data, shuffle=False, test_size=0.2)


print("Средний объём электропотребления в день:", test['PJME_MW'].mean())

pred_median = np.ones(test.shape) * train['PJME_MW'].median()
print("MAE:", mean_absolute_error(test ,pred_median))

2. Оцените модель вторым способом — предыдущим значением ряда. Предскажите дневной объём электропотребления и найдите для этого прогноза значение MAE.
В прекоде указан средний объём электропотребления, чтобы вы смогли соотнести его со значением метрики MAE.
Напечатайте на экране значения среднего объёма электропотребления и метрики MAE (уже в прекоде).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data.resample('1D').sum()

train, test = train_test_split(data, shuffle=False, test_size=0.2)

print("Средний объём электропотребления в день:", test['PJME_MW'].mean())

pred_previous = test.shift()
# заполняем первое значение
pred_previous.iloc[0] = train.iloc[-1]
print("MAE:", mean_absolute_error(test, pred_previous))

#Создание признаков

1. Напишите функцию make_features() (англ. «создать признаки»), чтобы прибавить к таблице четыре новых календарных признака: год, месяц, день и день недели.
Имена столбцов должны быть такие: 'year', 'month', 'day', 'dayofweek'.
Примените функцию к таблице и напечатайте на экране её первые пять строк (уже в прекоде).

import pandas as pd
import numpy as np


data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data.resample('1D').sum()

def make_features(data):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek

make_features(data)
print(data.head())

2. Вычислите отстающие значения. В функцию make_features() добавьте новый аргумент max_lag, который задаст максимальный размер отставания.
Новые признаки назовите: 'lag_1', 'lag_2' — и до величины max_lag.
Примените функцию к таблице и напечатайте на экране её первые пять строк (уже в прекоде).

import pandas as pd
import numpy as np


data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data.resample('1D').sum()

def make_features(data, max_lag):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['PJME_MW'].shift(lag)


make_features(data, 4)
print(data.head())


3. Вычислите скользящее среднее и добавьте его как признак 'rolling_mean'.
В функцию make_features() добавьте новый аргумент rolling_mean_size, который задаст ширину окна.
Текущее значение ряда для расчёта скользящего среднего применять нельзя.
Примените функцию к таблице и напечатайте на экране её первые пять строк (уже в прекоде).

import pandas as pd
import numpy as np


data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data.resample('1D').sum()

def make_features(data, max_lag, rolling_mean_size):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['PJME_MW'].shift(lag)

    data['rolling_mean'] = data['PJME_MW'].shift().rolling(4).mean()

make_features(data, 4, 4)
print(data.head())

#Обучение модели

1. Разбейте датасет о потреблении электроэнергии на обучающую и тестовую выборки в соотношении 4:1. Вам нужны данные за всё время. Из обучающей выборки удалите строки с пропусками.
Напечатайте на экране размеры обучающей и тестовой выборки (уже в прекоде).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data.resample('1D').sum()

def make_features(data, max_lag, rolling_mean_size):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['PJME_MW'].shift(lag)

    data['rolling_mean'] = data['PJME_MW'].shift().rolling(rolling_mean_size).mean()

# мы выбрали произвольные значения аргументов
make_features(data, 1, 1)

train, test = train_test_split(data, shuffle=False, test_size=0.2)
train = train.dropna() 

print(train.shape)
print(test.shape)

2. В выборке выделите признаки и целевой признак. На них обучите линейную регрессию и сохраните её в переменной model.
Затем напечатайте на экране значения MAE для обучающей и тестовой выборок (уже в прекоде).
Подберите аргументы функции make_features() так, чтобы значение MAE на тестовой выборке было не больше 37 000.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data.resample('1D').sum()

def make_features(data, max_lag, rolling_mean_size):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['PJME_MW'].shift(lag)

    data['rolling_mean'] = data['PJME_MW'].shift().rolling(rolling_mean_size).mean()


make_features(data, 1, 1)

train, test = train_test_split(data, shuffle=False, test_size=0.2)
train = train.dropna()

features_train  = train.drop('PJME_MW', axis=1)
target_train = train[['PJME_MW']]
features_test = test.drop('PJME_MW', axis=1)
target_test = test[['PJME_MW']]

model = LinearRegression()

model1 = model.fit(features_train, target_train) 
predictions1 = model1.predict(features_test) # получим предсказания модели
result1 = mean_absolute_error(target_test,predictions1)

model2 = model.fit(features_test, target_test) 
predictions2 = model2.predict(features_train) # получим предсказания модели
result2 = mean_absolute_error(target_train,predictions2)                          


print("MAE обучающей выборки:", result1)
print("MAE тестовой выборки: ", result2)
