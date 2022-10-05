#Что относится к задаче кластеризации? Выберите правильные ответы:
#Выделение объектов на изображении.
#Определение сегментов товаров в интернет-магазине.


Алгоритм k-средних
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

1. В документации sklearn найдите метод k-средних и импортируйте его. Обучите модель для трёх кластеров пользователей и заданного параметра random_state=12345. 
Напечатайте на экране значения центроидов полученных кластеров.

import pandas as pd
import warnings
from sklearn.cluster import KMeans

data = pd.read_csv('/datasets/segments.csv')

# Обучение модели
model = KMeans(n_clusters=3, random_state=12345).fit(data)


print("Центроиды кластеров:")
print(model.cluster_centers_)

2. В документации метода k-средних найдите, как модели можно передать начальные центроиды.
К прекоду добавьте обучение модели с начальными центроидами, заданными в переменной centers. Выведите на экран:
    центроиды кластеров для модели из прошлого задания (уже в прекоде),
    центроиды кластеров для модели с начальными центроидами.
    
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv('/datasets/segments.csv')
centers = np.array([[20, 80, 8], [50, 20, 5], [20, 30, 10]])

model = KMeans(n_clusters=3, random_state=12345)
model.fit(data)

print("Центроиды кластеров:")
print(model.cluster_centers_)

# Обучение модели с начальными центроидами
model =  KMeans(n_clusters=3, random_state=12345, init=centers)
model.fit(data)
# < напишите код здесь >

print("Центроиды кластеров для модели с начальными центроидами:")
print(model.cluster_centers_)

Задача.
В документации метода sklearn.cluster.KMeans найдите атрибут, отвечающий за целевую функцию. 
Добавьте к коду из предыдущего урока подсчёт этой функции для двух моделей: без начальных центроидов и с ними. Напечатайте на экране значения целевой функции для обеих моделей.    

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


data = pd.read_csv('/datasets/segments.csv')
centers = np.array([[20, 80, 8], [50, 20, 5], [20, 30, 10]])

model = KMeans(n_clusters=3, random_state=12345)
model.fit(data)

print("Целевая функция:")
print(model.inertia_)

model = KMeans(n_clusters=3, init=centers, random_state=12345)
model.fit(data)

print("Целевая функция модели с начальными центроидами:")
print(model.inertia_)


Локальный минимум
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv('/datasets/segments.csv')
centers = np.array([[20, 80, 8], [50, 20, 5], [20, 30, 10]])

model = KMeans(n_clusters=3, random_state=12345)
model.fit(data)

print("Целевая функция:")
print(model.inertia_)

model = KMeans(n_clusters=3, init=centers, random_state=12345)
model.fit(data)

print("Целевая функция модели с начальными центроидами:")
print(model.inertia_)


#Визуализация
Обучите модель с начальными центроидами centers. Постройте диаграмму pairplot с заливкой по кластерам и центроидами полученных кластеров. 
Начальные центроиды добавьте отдельным слоем без заливки.

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

data = pd.read_csv('https://code.s3.yandex.net/datasets/segments.csv')
centers = np.array([[20, 80, 8], [50, 20, 5], [20, 30, 10]])

model = KMeans(n_clusters=3, init=centers, random_state=12345)
model.fit(data)
centroids = pd.DataFrame(model.cluster_centers_, columns=data.columns)
data['label'] = model.labels_.astype(str)
centroids['label'] = ['0 centroid', '1 centroid', '2 centroid']
data_all = pd.concat([data, centroids], ignore_index=True)

pairgrid = sns.pairplot(data_all, hue='label', diag_kind='hist')
    # Дополнительный слой для центроидов
#pairgrid.map_offdiag(func=sns.scatterplot, s=200, marker='*', color='red')
# Сформируйте таблицу для дополнительного слоя
centroids_init = pd.DataFrame([[20, 80, 8], [50, 20, 5], [20, 30, 10]], \
                             columns=data.drop(columns=['label']).columns)
centroids_init['label'] = 4
pairgrid.data = centroids_init
pairgrid.map_offdiag(func=sns.scatterplot, s=200, marker='*', color='red',palette='flag')


#Оптимальное число кластеров

1. Выведите на экран значения целевой функции для количества кластеров — от 1 до 7. При обучении примените параметр random_state=12345.

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv('/datasets/segments.csv')

K = range(1, 8)
for k in K:
    model = KMeans(n_clusters=k, random_state=12345)
    model.fit(data)
    print('Число кластеров:', k) 
    print('Значение целевой функции', model.inertia_)
    
2. Обучите модель для четырёх кластеров. 
Центроиды укажите так: ['0 centroid', '1 centroid', '2 centroid', '3 centroid']. Постройте диаграмму pairplot с полученными центроидами и заливкой для модели. При обучении примените параметр random_state=12345.  

import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv('/datasets/segments.csv')

# Обучение модели для 4-х кластеров
model = KMeans(n_clusters=4, random_state=12345)
model.fit(data)

centroids = pd.DataFrame(model.cluster_centers_, columns=data.columns)
# < напишите код здесь >
data['label'] = model.labels_.astype(str)
centroids['label'] =['0 centroid', '1 centroid', '2 centroid', '3 centroid']
# < напишите код здесь >
data_all = pd.concat([data, centroids], ignore_index=True)

# Построение графика
sns.pairplot(data_all, hue='label', diag_kind='hist')

3. Обучите модели для трёх и четырёх кластеров. Выведите на экран округлённые центроиды полученных моделей. При обучении примените параметр random_state=12345.
  
  
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('/datasets/segments.csv')

# Обучение модели для 3-х кластеров
model = KMeans(n_clusters=3, random_state=12345)
model.fit(data)

print("Типичные пользователи сегментов для 3-х кластеров:")
print(model.cluster_centers_.round())

model = KMeans(n_clusters=4, random_state=12345)
model.fit(data)

print("Типичные пользователи сегментов для 4-х кластеров:")
print(model.cluster_centers_.round())

    
#Поиск структуры в данных

1. Постройте график метода локтя для количества кластеров от 1 до 10. При обучении модели примените параметр random_state=12345.
Составьте список distortion значений целевой функции для количества кластеров от 1 до 10. При этом используйте параметр random_state=12345.
Для полученных значений целевой функции постройте график метода локтя размером 12 на 8. Ось X назовите «Число кластеров», а ось Y — «Значение целевой функции».

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('/datasets/cars.csv')

distortion = []
K = range(1,11)
for k in K:
    model = KMeans(n_clusters=k, random_state=12345)
    model.fit(data)
    distortion.append(model.inertia_)
plt.figure(figsize=(12, 8))
plt.plot(K, distortion, 'bx-')
plt.xlabel('Число кластеров')
plt.ylabel('Значение целевой функции')
plt.show() 


2. Постройте диаграмму pairplot для модели с тремя кластерами без отмеченных центроидов. При обучении модели примените параметр random_state=12345.
Из-за особенностей версий seaborn нужно указать список признаков в функции pairplot(): vars=data.columns[:-1]. Последний признак — это номер кластера, его отображать не надо.


import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns

data = pd.read_csv('/datasets/cars.csv')
model = KMeans(n_clusters=3, random_state=12345)
model.fit(data)
data['label'] = model.labels_.astype(str)

sns.pairplot(data, hue='label', vars=data.columns[:-1], diag_kind='hist')

3. Постройте диаграмму pairplot с заливкой по столбцу brand. Обучите модель с тремя кластерами на данных без столбца brand.
Добавьте на график полученные центроиды. При обучении модели примените параметр random_state=12345.

import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
 
data_full = pd.read_csv('/datasets/cars_label.csv')
 
data = data_full.drop(columns=['brand'])

# Обучение модели
model = KMeans(n_clusters=3, random_state=12345)
model.fit(data)
 
# Дополнительный слой для центроидов
centroids = pd.DataFrame(model.cluster_centers_, columns=data.columns)


#  Сформируйте в дополнительной таблице новый столбец 'brand' в качестве заглушки
centroids['brand'] = 6
 
# Построение графика

pairgrid = sns.pairplot(data_full, hue='brand', diag_kind='hist')
pairgrid.data = centroids
pairgrid.map_offdiag(func=sns.scatterplot, s=200, marker='*', palette='flag')

#Диаграмма размаха

Найдите в датасете аномалии по признаку 'Profit'. Из ящика с усами возьмите список аномалий и запишите результат в переменной outliers.
Отфильтруйте исходный датафрейм функцией isin() и сохраните список объектов с аномалиями в переменной df_outliers.
Выведите количество аномалий (уже в прекоде).

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/datasets/sales.csv')

boxplot = plt.boxplot(df['Profit'].values)
outliers = list(boxplot["fliers"][0].get_data()[1])
df_outliers = df[df["Profit"].isin(outliers)]

print("Количество аномалий: ", len(df_outliers))

#Изоляционный лес

Обучите модель изоляционного леса и вычислите количество аномалий по признакам:
    продаж df['Sales'];
    прибыли df['Profit'].
Определите, какие объекты — выбросы, и запишите их в переменной outliers.
Выведите длину списка (уже в прекоде).

import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv('/datasets/sales.csv')
data = df[['Sales', 'Profit']]
isolation_forest = IsolationForest(n_estimators=100)
isolation_forest.fit(data)
anomaly_scores = isolation_forest.decision_function(data)
estimator = isolation_forest.fit_predict(data)
outliers = list(estimator[estimator == -1])
print("Количество аномалий: ", len(outliers))


#KNN для поиска аномалий

Моделью KNN и изоляционным лесом найдите выбросы в данных с переменными 'Sales' и 'Profit'. Выясните, сколько аномалий совпало.
Напечатайте на экране два варианта количества выбросов и число совпавших аномалий. Формат вывода указан в прекоде.

import pandas as pd
from pyod.models.knn import KNN
from sklearn.ensemble import IsolationForest

df = pd.read_csv('/datasets/sales.csv')
data = df[['Sales', 'Profit']]

model_knn = KNN()
model_knn.fit(data)
estimation_knn = model_knn.fit_predict(data)
estimation_knn = estimation_knn == 1
outliers_knn = estimation_knn.sum()
print("Количество аномалий (KNN): ", outliers_knn)

model_iforest = IsolationForest(n_estimators=100, random_state=12345)
estimation_iforest = model_iforest.fit_predict(data)
estimation_iforest = estimation_iforest == -1
outliers_iforest = estimation_iforest.sum()
print("Количество аномалий (изоляционный лес): ", outliers_iforest)

print("Совпало: ", (estimation_knn & estimation_iforest).sum())
