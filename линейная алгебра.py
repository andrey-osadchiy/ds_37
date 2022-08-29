#Создание векторов

1. Создайте два вектора: первый содержит все ответы на вопрос о цене, второй — о качестве.
Напечатайте результат на экране (уже в прекоде).


import numpy as np
import pandas as pd

reviews_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
reviews = pd.DataFrame(reviews_values, columns=['Цена', 'Качество'])

price = reviews['Цена'].values
quality = reviews['Качество'].values
print("Цена: ", price)
print("Качество: ", quality)


2. Определите общее количество покупателей «НосиВипчик».
Найдите вектор с оценками клиента с индексом 4 (нумерация индексов начинается с нуля).
Напечатайте результат на экране (уже в прекоде).

import numpy as np
import pandas as pd

reviews_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
reviews = pd.DataFrame(reviews_values, columns=['Цена', 'Качество'])

visitors_count = len(reviews['Цена'].values)
visitor4 = reviews.loc[4].values
print("Количество покупателей:", visitors_count)
print("Покупатель 4:", visitor4)

3. У таблицы есть атрибут values — это двумерный массив.
Вызовом функции list() преобразуйте его в список векторов с оценками всех клиентов. Последняя строчка кода выведет результат.

import numpy as np
import pandas as pd

reviews_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
reviews = pd.DataFrame(reviews_values, columns=['Цена', 'Качество'])

vector_list = list(reviews.values)
print(vector_list)


#Изображение векторов

1. Изобразите стрелкой вектор [75, 15] из кейса об интернет-магазине одежды «НосиВипчик».

import numpy as np
import matplotlib.pyplot as plt

vector = np.array([75, 15])
plt.figure(figsize=(7, 7))
plt.axis([0, 100, 0, 100]) 
plt.arrow(0, 0,vector[0],vector[1],length_includes_head="True", color='b')
plt.xlabel('Цена')
plt.ylabel('Качество')
plt.grid(True)
plt.show()

2. Представьте все двумерные векторы с оценками клиентов интернет-магазина «НосиВипчик» в виде точек на плоскости.
Какие клиенты с какого агрегатора пришли?


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

reviews_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
reviews = pd.DataFrame(reviews_values, columns=['Цена', 'Качество'])

plt.figure(figsize=(7, 7))
plt.axis([0, 100, 0, 100])
price = reviews['Цена'].values
quality = reviews['Качество'].values
plt.plot(price,quality,'ro')
plt.xlabel('Цена')
plt.ylabel('Качество')
plt.grid(True)
plt.show()

3.Создайте два отдельных списка двумерных векторов с оценками клиентов, которые пришли с первого сайта-агрегатора (масс-маркет) и со второго (люкс).
Назовите переменные clients_1 и clients_2, а потом напечатайте их значения на экране.

import numpy as np
import pandas as pd

reviews_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
reviews = pd.DataFrame(reviews_values, columns=['Цена', 'Качество'])

clients_1 = []
clients_2 = []
for client in list(reviews.values):
    if (client[0] > 60 and client[1] < 40):
        clients_1.append(client)
    elif (client[0]< 40 and client[1] > 60):
        clients_2.append(client)
        
print("Оценки пришедших с первого агрегатора:", clients_1)
print("Оценки пришедших со второго агрегатора:", clients_2)


#Сложение и вычитание векторов

1. Выберите из каждой таблицы столбец с количеством и преобразуйте в вектор. Напечатайте результат на экране (уже в прекоде).
import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = ['Чехол силиконовый для iPhone 8',
          'Чехол кожаный для iPhone 8',
          'Чехол силиконовый для iPhone XS',
          'Чехол кожаный для iPhone XS',
          'Чехол силиконовый для iPhone XS Max',
          'Чехол кожаный для iPhone XS Max',
          'Чехол силиконовый для iPhone 11',
          'Чехол кожаный для iPhone 11',
          'Чехол силиконовый для iPhone 11 Pro',
          'Чехол кожаный для iPhone 11 Pro',
         ]
stocks_1 = pd.DataFrame({'Количество' : quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Количество' : quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Количество'].values
vector_of_quantity_2 =  stocks_2['Количество'].values
print("Запасы первого магазина:", vector_of_quantity_1,
      "\nЗапасы второго магазина:",vector_of_quantity_2)
      
2. Вычислите вектор со складским запасом объединённого интернет-магазина «АйДаЧехлоФон».
Напечатайте результат на экране (уже в прекоде).     

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = ['Чехол силиконовый для iPhone 8',
          'Чехол кожаный для iPhone 8',
          'Чехол силиконовый для iPhone XS',
          'Чехол кожаный для iPhone XS',
          'Чехол силиконовый для iPhone XS Max',
          'Чехол кожаный для iPhone XS Max',
          'Чехол силиконовый для iPhone 11',
          'Чехол кожаный для iPhone 11',
          'Чехол силиконовый для iPhone 11 Pro',
          'Чехол кожаный для iPhone 11 Pro',
         ]
stocks_1 = pd.DataFrame({'Количество' : quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Количество' : quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Количество'].values
vector_of_quantity_2 = stocks_2['Количество'].values

vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2
print(vector_of_quantity_united)

3. Постройте таблицу с ассортиментом объединённого магазина. Напечатайте результат на экране (уже в прекоде).

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = ['Чехол силиконовый для iPhone 8',
          'Чехол кожаный для iPhone 8',
          'Чехол силиконовый для iPhone XS',
          'Чехол кожаный для iPhone XS',
          'Чехол силиконовый для iPhone XS Max',
          'Чехол кожаный для iPhone XS Max',
          'Чехол силиконовый для iPhone 11',
          'Чехол кожаный для iPhone 11',
          'Чехол силиконовый для iPhone 11 Pro',
          'Чехол кожаный для iPhone 11 Pro',
         ]
stocks_1 = pd.DataFrame({'Количество' : quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Количество' : quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Количество'].values
vector_of_quantity_2 = stocks_2['Количество'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2

stocks_united = pd.DataFrame({'Количество' : vector_of_quantity_united}, index=models)
print(stocks_united)

#Умножение вектора на число

1. Выберите из таблицы столбец с ценой и преобразуйте его в числовой вектор.
Напечатайте результат на экране (уже в прекоде).

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = ['Чехол силиконовый для iPhone 8',
          'Чехол кожаный для iPhone 8',
          'Чехол силиконовый для iPhone XS',
          'Чехол кожаный для iPhone XS',
          'Чехол силиконовый для iPhone XS Max',
          'Чехол кожаный для iPhone XS Max',
          'Чехол силиконовый для iPhone 11',
          'Чехол кожаный для iPhone 11',
          'Чехол силиконовый для iPhone 11 Pro',
          'Чехол кожаный для iPhone 11 Pro',
         ]
stocks_1 = pd.DataFrame({'Количество' : quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Количество' : quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Количество'].values
vector_of_quantity_2 = stocks_2['Количество'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2

stocks_united = pd.DataFrame({'Количество' : vector_of_quantity_united}, index=models
                            )
stocks_united['Цена'] = [3000, 2100, 3200, 2200, 1800, 1700, 3800, 1200, 2300, 2900]

price_united = stocks_united['Цена'].values 
print(price_united)


2. «АйДаЧехлоФон» объявил о 10%-й скидке на весь объединённый ассортимент.
С учётом этой акции вычислите вектор с ценами. Напечатайте результат на экране (уже в прекоде).

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = ['Чехол силиконовый для iPhone 8',
          'Чехол кожаный для iPhone 8',
          'Чехол силиконовый для iPhone XS',
          'Чехол кожаный для iPhone XS',
          'Чехол силиконовый для iPhone XS Max',
          'Чехол кожаный для iPhone XS Max',
          'Чехол силиконовый для iPhone 11',
          'Чехол кожаный для iPhone 11',
          'Чехол силиконовый для iPhone 11 Pro',
          'Чехол кожаный для iPhone 11 Pro',
         ]
stocks_1 = pd.DataFrame({'Количество' : quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Количество' : quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Количество'].values
vector_of_quantity_2 = stocks_2['Количество'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2

stocks_united = pd.DataFrame({'Количество' : vector_of_quantity_united}, index=models
                            )
stocks_united['Цена'] = [3000, 2100, 3200, 2200, 1800, 1700, 3800, 1200, 2300, 2900]

price_united = stocks_united['Цена'].values

price_discount_10 = price_united * 0.9
stocks_united['Цена со скидкой 10%'] = price_discount_10.astype(int)
print(stocks_united)


3. После месяца скидок «АйДаЧехлоФон» поднял цены на 10%. Создайте прайс-лист с учётом повышения.
Напечатайте результат на экране (уже в прекоде).

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = ['Чехол силиконовый для iPhone 8',
          'Чехол кожаный для iPhone 8',
          'Чехол силиконовый для iPhone XS',
          'Чехол кожаный для iPhone XS',
          'Чехол силиконовый для iPhone XS Max',
          'Чехол кожаный для iPhone XS Max',
          'Чехол силиконовый для iPhone 11',
          'Чехол кожаный для iPhone 11',
          'Чехол силиконовый для iPhone 11 Pro',
          'Чехол кожаный для iPhone 11 Pro',
         ]
stocks_1 = pd.DataFrame({'Количество' : quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Количество' : quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Количество'].values
vector_of_quantity_2 = stocks_2['Количество'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2

stocks_united = pd.DataFrame({'Количество' : vector_of_quantity_united}, index=models
                            )
stocks_united['Цена'] = [3000, 2100, 3200, 2200, 1800, 1700, 3800, 1200, 2300, 2900]

price_united = stocks_united['Цена'].values
price_discount_10 = price_united * 0.9
stocks_united['Цена со скидкой 10%'] = price_discount_10.astype(int)

price_no_discount =  price_discount_10 * 1.1
stocks_united['Повышенная на 10% цена'] = price_no_discount.astype(int)
print(stocks_united)


#Среднее значение векторов

1. Вычислите среднее значение оценки удовлетворённости качеством.
А функция print(average_quality_rev) напечатает результат на экране.

import numpy as np
import pandas as pd

reviews_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
reviews = pd.DataFrame(reviews_values, columns=['Цена', 'Качество'])

quality  = reviews['Качество'].values

average_quality_rev =  sum(quality) / len(quality)
print(average_quality_rev)

2. Мы добавили в код расчёт средней цены и сохранили её в переменной average_price_rev.
Теперь сохраните в массив average_rev:
среднюю оценку цены,
среднюю оценку качества.
Функция print() выведет массив на экран.

import numpy as np
import pandas as pd

reviews_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
reviews = pd.DataFrame(reviews_values, columns=['Цена', 'Качество'])

price = reviews['Цена'].values
sum_prices = sum(price)
average_price_rev = sum(price) / len(price)

quality = reviews['Качество'].values
average_quality_rev = sum(quality) / len(quality)

average_rev = np.array([average_price_rev,average_quality_rev])
print(average_rev)


3. Обозначьте на координатной плоскости найденное среднее значение.
Можно ли считать получившийся вектор оценками типичного покупателя? Напечатайте результат на экране.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

reviews_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
reviews = pd.DataFrame(reviews_values, columns=['Цена', 'Качество'])

price = reviews['Цена'].values
sum_prices = sum(price)
average_price_rev = sum(price) / len(price)

quality = reviews['Качество'].values
average_quality_rev = sum(quality) / len(quality)
average_rev = np.array([average_price_rev, average_quality_rev])

plt.figure(figsize=(7, 7))
plt.axis([0, 100, 0, 100])
plt.plot(average_price_rev,average_quality_rev , 'mo', markersize=15)
plt.plot(price, quality, 'ro')
plt.xlabel('Цена')
plt.ylabel('Качество')
plt.grid(True)
plt.title("Распределение оценок и среднее значение по всей выборке")
plt.show()

4. Вычислите средние оценки отдельно для клиентов, которые пришли с первого сайта-агрегатора (масс-маркет) и со второго (люкс).
Результат должен появиться на экране.

import numpy as np
import pandas as pd

reviews_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
reviews = pd.DataFrame(reviews_values, columns=['Цена', 'Качество'])

clients_1 = []
clients_2 = []
for client in list(reviews.values):
    if client[0] < 40 and client[1] > 60:
        clients_2.append(client)
    else:
        clients_1.append(client)

average_client_1 = sum(clients_1) / len(clients_1)
print('Средняя оценка пришедших с первого агрегатора: ', average_client_1)

average_client_2 = sum(clients_2) / len(clients_2)
print('Средняя оценка пришедших со второго агрегатора: ', average_client_2)

5. Изобразите найденные значения средних на диаграмме с оценками отдельных клиентов.
Можно ли считать каждое найденное среднее оценкой типичного клиента этой группы? Выведите результат на экран.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

reviews_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
reviews = pd.DataFrame(reviews_values, columns=['Цена', 'Качество'])
price = reviews['Цена'].values
quality = reviews['Качество'].values

clients_1 = []
clients_2 = []
for client in list(reviews.values):
    if client[0] < 40 and client[1] > 60:
        clients_2.append(client)
    else:
        clients_1.append(client)

average_client_1 = sum(clients_1)/len(clients_1)

average_client_2 = sum(clients_2)/len(clients_2)

plt.figure(figsize=(7, 7))
plt.axis([0, 100, 0, 100])

# изображаем среднее для группы 1
# 'b' — синий цвет (англ. blue)
plt.plot(average_client_1[0],average_client_1[1], 
         'bo', markersize=15)

# изображаем среднее для группы 2
# 'g' — зелёный цвет (англ. green)
plt.plot(average_client_2[0],average_client_2[1],
         'go', markersize=15)
plt.plot(price, quality, 'ro')
plt.xlabel('Цена')
plt.ylabel('Качество')
plt.grid(True)
plt.title("Распределение оценок и среднее значение для каждой группы")
plt.show()

#Векторизованные функции
Задача.
Напишите функцию logistic_transform(), выполняющую логистическое преобразование. 
Примените её ко всем элементам массива. Напечатайте результат на экране (уже в прекоде).

import numpy as np

def logistic_transform(values):
    return 1/ (1 + np.exp(-values))

our_values = np.array([-20, 0, 0.5, 80, -1])
print(logistic_transform(our_values))

#Векторизация метрик
1. Напишите функцию для вычисления MAE с применением метода mean(). 
Посчитайте MAE и напечатайте результат на экране (уже в прекоде).

import numpy as np

def mae(target, predictions):
    return sum(np.abs(target -predictions))/len(target)

target = np.array([0.9, 1.2, 1.4, 1.5, 1.9, 2.0])
predictions = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
print(mae(target, predictions))

2. Рассчитайте RMSE по этой формуле:

import numpy as np

def rmse(target, predictions):
    return (sum((target - predictions)**2)/len(target))**0.5

target = np.array([0.9, 1.2, 1.4, 1.5, 1.9, 2.0])
predictions = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
print(rmse(target, predictions))

# Скалярное произведение векторов
Задача
Найдите стоимость всех товаров в первом и втором магазинах сети «Я вас услышал».
Сохраните их в переменных stocks1_cost (англ. «стоимость запасов на складе») и stocks2_cost.
Найдите суммарную стоимость товаров двух магазинов сети и сохраните её в переменной total_cost (англ. «общая стоимость»).
Напечатайте результат на экране (уже в прекоде).
import numpy as np
import pandas as pd

shop1_price = [20990, 11990 , 5390, 3190, 1990, 10990, 5999, 2290, 8111 , 3290]
shop1_quantity = [19, 11, 8, 15, 23, 7, 14, 9, 10, 4]

shop2_price = [20990, 12490, 4290, 2790, 2390, 10990, 4990, 2490, 8990, 3290]
shop2_quantity = [10, 16, 20, 9, 18, 12, 10, 11, 18, 22]

models = ['Apple AirPods Pro',
          'Apple AirPods MV7N2RU/A',
          'JBL Tune 120TWS', 
          'JBL TUNE 500BT',
          'JBL JR300BT', 
          'Huawei Freebuds 3',
          'Philips TWS SHB2505',
          'Sony WH-CH500',
          'Sony WF-SP700N',
          'Sony WI-XB400',
         ]
stocks1 = pd.DataFrame({'Цена':shop1_price, 
                        'Количество':shop1_quantity}, index=models)
stocks2 = pd.DataFrame({'Цена':shop2_price, 
                        'Количество':shop2_quantity}, index=models)

stocks1_price = stocks1['Цена'].values 
stocks1_quantity = stocks1['Количество'].values 

stocks2_price = stocks2['Цена'].values # < напишите код здесь >
stocks2_quantity = stocks2['Количество'].values# < напишите код здесь >

# общая стоимость товаров в магазине 1
stocks1_cost = sum(stocks1_price * stocks1_quantity)# < напишите код здесь >

# общая стоимость товаров в магазине 2
stocks2_cost = sum(stocks2_price * stocks2_quantity) # < напишите код здесь >

total_cost = stocks1_cost + stocks2_cost # < напишите код здесь >

print('Общая стоимость всех товаров в сети:', total_cost, 'руб.')

#Расстояние на плоскости
1. Постройте таблицу расстояний между населёнными пунктами и сохраните её в переменной distances.
Представьте данные как список списков. Каждая строка — это расстояние от одного населённого пункта до остальных.
Добавьте в таблицу distances названия всех сёл и деревень.
Обратите внимание, что здесь колонки называются не по правилам Python — кириллицей и с заглавной буквы.
Названия деревень на латинице выглядят нечитабельно, а код должен понимать не только автор, но и другие разработчики.
Функция print() выведет результат на экран.

import numpy as np
import pandas as pd
from scipy.spatial import distance


x_axis = np.array([0.0, 0.18078584, 9.32526599, 17.09628721,
                      4.69820241, 11.57529305, 11.31769349, 14.63378951])

y_axis  = np.array([0.0, 7.03050245, 9.06193657, 0.1718145,
                      5.1383203, 0.11069032, 3.27703365, 5.36870287])

shipments = np.array([5, 7, 4, 3, 5, 2, 1, 1])

village = ['Тетерье',  
           'Журавец', 
           'Корсунь', 
           'Берёзовка', 
           'Протасово',  
           'Трудки',  
           'Нижний Туровец',  
           'Вышний Туровец']

data = pd.DataFrame({'x_coordinates_km': x_axis,
                     'y_coordinates_km': y_axis, 
                     'deliveries': shipments}, index=village)

vectors = data[['x_coordinates_km', 'y_coordinates_km']].values

distances = []
for village_from in range(len(village)):
    row = []
    for village_to in range(len(village)):
        value = distance.euclidean(vectors[village_from],vectors[village_to])
        row.append(value)
    distances.append(row) 
distances_df = pd.DataFrame(distances, index=village, columns=village)
#print(distances_df)

2. Вы знаете, сколько заказов за неделю доставляют в каждую точку. 
Выберите оптимальный для склада компании «Дрон Горыныч» населённый пункт. 
Для этого найдите расстояние между пунктами, удвойте его (полёты туда и обратно) и умножьте на еженедельное количество доставок. 
Сохраните результат в списке shipping_in_week.
Выберите населенный пункт с наименьшей суммарной дистанцией до соседей. 
Выведите результат на экран
import numpy as np
import pandas as pd
from scipy.spatial import distance

x_axis = np.array([0.0, 7.03050245, 9.06193657, 0.1718145,
                      5.1383203, 0.11069032, 3.27703365, 5.36870287])
y_axis = np.array([0., 0.18078584, 9.32526599, 17.09628721,
                      4.69820241, 11.57529305, 11.31769349, 14.63378951])
shipments = np.array([5, 7, 4, 3, 5, 2, 1, 1])

village = ['Тетерье',  
           'Журавец', 
           'Корсунь', 
           'Берёзовка', 
           'Протасово',  
           'Трудки',  
           'Нижний Туровец',  
           'Вышний Туровец']

data = pd.DataFrame({'x_coordinates_km': x_axis,
                     'y_coordinates_km': y_axis, 
                     'deliveries': shipments}, index=village)

vectors = data[['x_coordinates_km', 'y_coordinates_km']].values

distances = []
for village_from in range(len(village)):
    row = []
    for village_to in range(len(village)):
        value = distance.euclidean(vectors[village_from], vectors[village_to])
        row.append(value)
    distances.append(row)

shipping_in_week = []
for i in range((len(shipments))):
    value = 2 * np.dot(np.array(distances[i]), shipments)
    shipping_in_week.append(value)
    
shipping_in_week_df = pd.DataFrame({'distance': shipping_in_week}, index=village)

print(shipping_in_week_df)

print()
print('Населённый пункт для склада:', 
    shipping_in_week_df['distance'].idxmin())
    
1. Напишите функцию для вычисления манхэттенского расстояния — manhattan_distance(). 
На вход она принимает два вектора, а возвращает расстояние. Решите задачу, создав векторизованную функцию. К циклам обращаться нельзя.

import numpy as np

def manhattan_distance(first, second):
    return np.abs(first - second).sum() 

first = np.array([3, 11])
second = np.array([1, 6])

print(manhattan_distance(first, second))


2.
Из трёх свободных такси найдите ближайшее в нужном районе Манхэттена.
Заданы такие переменные:
avenues_df, в которой сохранён список авеню и их координат;
streets_df — список улиц и координат;
address — перекрёсток, на котором клиент вызывает такси;
taxies — места парковки такси.
Найдите список расстояний всех машин до нужного адреса и сохраните результат в переменной taxies_distances.
Самостоятельно найдите в документации нужную функцию в библиотеке SciPy и импортируйте её.
Определите порядковый номер ближайшего такси. Напечатайте на экране название перекрёстка, где оно стоит (уже в прекоде).

import numpy as np
import pandas as pd
from scipy.spatial import distance

avenues_df = pd.DataFrame([0, 153, 307, 524], index=['Park', 'Lexington', '3rd', '2nd'])
streets_df = pd.DataFrame([0, 81, 159, 240, 324], index=['76', '75', '74', '73', '72'])

address = ['Lexington', '74']
taxies = [
    ['Park', '72'],
    ['2nd', '75'],
    ['3rd', '76'],
]

address_vector = np.array([avenues_df.loc[address[0]], streets_df.loc[address[1]]])
taxi_distances = []
for i in range(len(taxies)):
    taxi_vector = np.array(
        [avenues_df.loc[taxies[i][0]], streets_df.loc[taxies[i][1]]])
    value =distance.cityblock(address_vector,taxi_vector) 
    taxi_distances.append(value)
                     
index = np.argmin(taxi_distances)# < напишите код здесь >
print(taxies[index])


Расстояния в многомерном пространстве
1.Сохраните векторы квартир с индексами 3 и 11 в переменных vector_first («первый вектор») и vector_second («второй вектор»).
Вычислите между ними евклидово и манхэттенское расстояния.
Напечатайте их значения на экране (уже в прекоде).
import pandas as pd
from scipy.spatial import distance

columns = ['комнаты', 'пл. общая', 'кухня', 'пл. жилая', 'этаж', 'всего этажей']
realty = [
    [1, 38.5, 6.9, 18.9, 3, 5],
    [1, 38.0, 8.5, 19.2, 9, 17],
    [1, 34.7, 10.3, 19.8, 1, 9],
    [1, 45.9, 11.1, 17.5, 11, 23],
    [1, 42.4, 10.0, 19.9, 6, 14],
    [1, 46.0, 10.2, 20.5, 3, 12],
    [2, 77.7, 13.2, 39.3, 3, 17],
    [2, 69.8, 11.1, 31.4, 12, 23],
    [2, 78.2, 19.4, 33.2, 4, 9],
    [2, 55.5, 7.8, 29.6, 1, 25],
    [2, 74.3, 16.0, 34.2, 14, 17],
    [2, 78.3, 12.3, 42.6, 23, 23],
    [2, 74.0, 18.1, 49.0, 8, 9],
    [2, 91.4, 20.1, 60.4, 2, 10],
    [3, 85.0, 17.8, 56.1, 14, 14],
    [3, 79.8, 9.8, 44.8, 9, 10],
    [3, 72.0, 10.2, 37.3, 7, 9],
    [3, 95.3, 11.0, 51.5, 15, 23],
    [3, 69.3, 8.5, 39.3, 4, 9],
    [3, 89.8, 11.2, 58.2, 24, 25],
]

df_realty = pd.DataFrame(realty, columns=columns)

vector_first = df_realty.loc[3].values
vector_second = df_realty.loc[11].values

print("Евклидово расстояние:", distance.euclidean(vector_first, vector_second))
print("Манхэттенское расстояние:", distance.cityblock(vector_first, vector_second))

2.Клиенту понравилась квартира с индексом 12. Найдите к ней ближайшую по евклидовому расстоянию.
Создайте список с расстояниями всех векторов до вектора с номером 12.
Вычислите индекс наиболее похожего объекта и сохраните в переменной best_index.
Напечатайте результат на экране (уже в прекоде).

import numpy as np
import pandas as pd
from scipy.spatial import distance

columns = ['комнаты', 'пл. общая', 'кухня', 'пл. жилая', 'этаж', 'всего этажей']
realty = [
    [1, 38.5, 6.9, 18.9, 3, 5],
    [1, 38.0, 8.5, 19.2, 9, 17],
    [1, 34.7, 10.3, 19.8, 1, 9],
    [1, 45.9, 11.1, 17.5, 11, 23],
    [1, 42.4, 10.0, 19.9, 6, 14],
    [1, 46.0, 10.2, 20.5, 3, 12],
    [2, 77.7, 13.2, 39.3, 3, 17],
    [2, 69.8, 11.1, 31.4, 12, 23],
    [2, 78.2, 19.4, 33.2, 4, 9],
    [2, 55.5, 7.8, 29.6, 1, 25],
    [2, 74.3, 16.0, 34.2, 14, 17],
    [2, 78.3, 12.3, 42.6, 23, 23],
    [2, 74.0, 18.1, 49.0, 8, 9],
    [2, 91.4, 20.1, 60.4, 2, 10],
    [3, 85.0, 17.8, 56.1, 14, 14],
    [3, 79.8, 9.8, 44.8, 9, 10],
    [3, 72.0, 10.2, 37.3, 7, 9],
    [3, 95.3, 11.0, 51.5, 15, 23],
    [3, 69.3, 8.5, 39.3, 4, 9],
    [3, 89.8, 11.2, 58.2, 24, 25],
]

df_realty = pd.DataFrame(realty, columns=columns)

# англ. индекс понравившегося объекта
preference_index = 12
preference_vector = df_realty.loc[preference_index].values

distances = []

for i in range(len(df_realty)):
    vector = np.array(df_realty.loc[i])
    values = distance.euclidean(preference_vector, vector)
    distances.append(values)

best_index  = np.array(distances).argsort()[1] 

print("Индекс наиболее похожей квартиры:", best_index)

Метод ближайших соседей

Напишите функцию nearest_neighbor_predict() (англ. «предсказать методом ближайшего соседа»).
На вход она принимает три параметра:
    признаки обучающей выборки (train_features),
    целевой признак обучающей выборки (train_target),
    признаки нового объекта (new_features).
Функция возвращает предсказание целевого признака для нового объекта (new_features) методом ближайшего соседа.
Запустите метод на новом объекте new_apartment — предскажите, нужен ли там кондиционер. 
Напечатайте результат на экране (уже в прекоде).


import numpy as np
import pandas as pd
from scipy.spatial import distance

columns = ['комнаты', 'площадь', 'кухня', 'пл. жилая', 'этаж', 'всего этажей', 'кондиционер']

df_train = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 1],
    [1, 38.0, 8.5, 19.2, 9, 17, 0],
    [1, 34.7, 10.3, 19.8, 1, 9, 0],
    [1, 45.9, 11.1, 17.5, 11, 23, 1],
    [1, 42.4, 10.0, 19.9, 6, 14, 0],
    [1, 46.0, 10.2, 20.5, 3, 12, 1],
    [2, 77.7, 13.2, 39.3, 3, 17, 1],
    [2, 69.8, 11.1, 31.4, 12, 23, 0],
    [2, 78.2, 19.4, 33.2, 4, 9, 0],
    [2, 55.5, 7.8, 29.6, 1, 25, 1],
    [2, 74.3, 16.0, 34.2, 14, 17, 1],
    [2, 78.3, 12.3, 42.6, 23, 23, 0],
    [2, 74.0, 18.1, 49.0, 8, 9, 0],
    [2, 91.4, 20.1, 60.4, 2, 10, 0],
    [3, 85.0, 17.8, 56.1, 14, 14, 1],
    [3, 79.8, 9.8, 44.8, 9, 10, 0],
    [3, 72.0, 10.2, 37.3, 7, 9, 1],
    [3, 95.3, 11.0, 51.5, 15, 23, 1],
    [3, 69.3, 8.5, 39.3, 4, 9, 0],
    [3, 89.8, 11.2, 58.2, 24, 25, 0],
], columns=columns)


def nearest_neighbor_predict(train_features, train_target, new_features):
    # сделайте список с расстояниями до всех объектов
    # обучающей выборки
    distances = []
    for i in range(train_features.shape[0]):
        vector = train_features.loc[i].values
        distances.append(distance.euclidean(new_features, vector))
    best_index = np.array(distances).argmin()
    return train_target.loc[best_index]


train_features = df_train.drop('кондиционер', axis=1)
train_target = df_train['кондиционер']
new_apartment = np.array([2, 51.0, 8.2, 35.9, 5, 5])
prediction = nearest_neighbor_predict(train_features, train_target, new_apartment)
print(prediction)

Создание класса модели
1. Создайте класс NearestNeighborClassificator для модели классификации методом ближайших соседей. 
В этом задании будет только обучение, а в следующем — предсказание.
Добавьте в класс метод fit(). Для метода ближайшего соседа обучение модели — 
это запоминание всей обучающей выборки. 
В ней predict() будет искать ближайший объект.
Сохраните:
признаки обучающей выборки в атрибуте self.features_train;
целевой признак — self.target_train.
Ничего страшного, если атрибуты называются так же, как параметры.
Обучите модель, напечатайте на экране её атрибуты (уже в прекоде).

import pandas as pd
from scipy.spatial import distance

columns = ['комнаты', 'площадь', 'кухня', 'пл. жилая', 'этаж', 'всего этажей', 'кондиционер']

df_train = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 1],
    [1, 38.0, 8.5, 19.2, 9, 17, 0],
    [1, 34.7, 10.3, 19.8, 1, 9, 0],
    [1, 45.9, 11.1, 17.5, 11, 23, 1],
    [1, 42.4, 10.0, 19.9, 6, 14, 0],
    [1, 46.0, 10.2, 20.5, 3, 12, 1],
    [2, 77.7, 13.2, 39.3, 3, 17, 1],
    [2, 69.8, 11.1, 31.4, 12, 23, 0],
    [2, 78.2, 19.4, 33.2, 4, 9, 0],
    [2, 55.5, 7.8, 29.6, 1, 25, 1],
    [2, 74.3, 16.0, 34.2, 14, 17, 1],
    [2, 78.3, 12.3, 42.6, 23, 23, 0],
    [2, 74.0, 18.1, 49.0, 8, 9, 0],
    [2, 91.4, 20.1, 60.4, 2, 10, 0],
    [3, 85.0, 17.8, 56.1, 14, 14, 1],
    [3, 79.8, 9.8, 44.8, 9, 10, 0],
    [3, 72.0, 10.2, 37.3, 7, 9, 1],
    [3, 95.3, 11.0, 51.5, 15, 23, 1],
    [3, 69.3, 8.5, 39.3, 4, 9, 0],
    [3, 89.8, 11.2, 58.2, 24, 25, 0],
], columns=columns)
    

train_features = df_train.drop('кондиционер', axis=1)
train_target = df_train['кондиционер']

df_test = pd.DataFrame([
    [1, 36.5, 5.9, 17.9, 2, 7, 0],
    [2, 71.7, 12.2, 34.3, 5, 21, 1],
    [3, 88.0, 18.1, 58.2, 17, 17, 1],
], columns=columns)

test_features = df_test.drop('кондиционер', axis=1)

class NearestNeighborClassificator:
    def fit(self, features_train, target_train):
        self.features_train = features_train # < напишите код здесь >
        self.target_train = target_train# < напишите код здесь >

model = NearestNeighborClassificator()
model.fit(train_features, train_target)
print(model.features_train.head())
print(model.target_train.head())


2. К классу NearestNeighborClassificator добавьте метод predict(). 
Возьмите за основу уже написанную функцию nearest_neighbor_predict() — метод должен вернуть предсказание в виде Series.
Получите предсказания, есть кондиционеры или нет. Напечатайте результат на экране (уже в прекоде).

import numpy as np
import pandas as pd
from scipy.spatial import distance

columns = ['комнаты', 'площадь', 'кухня', 'пл. жилая', 'этаж', 'всего этажей', 'кондиционер']

df_train = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 1],
    [1, 38.0, 8.5, 19.2, 9, 17, 0],
    [1, 34.7, 10.3, 19.8, 1, 9, 0],
    [1, 45.9, 11.1, 17.5, 11, 23, 1],
    [1, 42.4, 10.0, 19.9, 6, 14, 0],
    [1, 46.0, 10.2, 20.5, 3, 12, 1],
    [2, 77.7, 13.2, 39.3, 3, 17, 1],
    [2, 69.8, 11.1, 31.4, 12, 23, 0],
    [2, 78.2, 19.4, 33.2, 4, 9, 0],
    [2, 55.5, 7.8, 29.6, 1, 25, 1],
    [2, 74.3, 16.0, 34.2, 14, 17, 1],
    [2, 78.3, 12.3, 42.6, 23, 23, 0],
    [2, 74.0, 18.1, 49.0, 8, 9, 0],
    [2, 91.4, 20.1, 60.4, 2, 10, 0],
    [3, 85.0, 17.8, 56.1, 14, 14, 1],
    [3, 79.8, 9.8, 44.8, 9, 10, 0],
    [3, 72.0, 10.2, 37.3, 7, 9, 1],
    [3, 95.3, 11.0, 51.5, 15, 23, 1],
    [3, 69.3, 8.5, 39.3, 4, 9, 0],
    [3, 89.8, 11.2, 58.2, 24, 25, 0],
], columns=columns)
    

train_features = df_train.drop('кондиционер', axis=1)
train_target = df_train['кондиционер']Поиск команды Practicum Students





1

O
2

4
3


Practicum Students






Slackbot
Привет! Я Slackbot.


Вы с нами! Здравствуйте!

Чтобы узнать больше об использовании Slack, нажмите  значок справки в правом верхнем углу окна приложения. (Или посетите Справочный центр в Интернете.)

Однако я не человек. Я бот (обычный бот, но у меня в виртуальном рукаве найдется несколько фокусов). Но мне приятно, что вы здесь!


Добро пожаловать в когорту 37DS
РАБОЧИЙ ПРОЦЕСС  11:59
Привет, теперь ты в 37 когорте профессии Специалист по Data Science (ds_37)
:ракета: Немного полезной информации для легкого старта:
По всем организационным вопросам ты можешь обращаться к куратору этой когорты — 
@Люба
Преподаватель по тренажеру (канал exerciser) - 
@Ющенко Артём
Преподаватель по проектам (канал projects) — 
@Анатолий Крестенко
Наставники (канал teamwork) - 
@Александр Ольферук
  и 
@Александр Пивовар
Проверь, что у тебя есть 5 каналов: info, library, exerciser, projects, teamwork. Прочитай все сообщения от куратора, это поможет быстрее разобраться с форматом общения в slack.
Также проверь, что тебе доступны следующие ресурсы - они понадобятся в учебе:
:перекидной_календарь: Календарь
:летающая_тарелка: Наш Я.Диск
:screenshot_17: Стартовая страничка в Notion
Не забудь отредактировать свой профиль Slack, указать Имя и Фамилию кириллицей и в поле What I Do добавить «Студент(ка) ds_37» :alex_r:
Поддержка: 
@Поддержка / Learning Support
 – там быстро помогут, если возникли вопросы по оплате курса или технические проблемы
Желаем тебе успехов в учёбе :praktikum:

Slackbot
  12:37
:машу_рукой: Вы с нами! Мы рады вас видеть. Это Slack, мессенджер для команд. Не знаете, с чего начать?
:почтовый_ящик_для_исходящих:
Рассылка для уточнения информации
ПРИЛОЖЕНИЕ  13:39
Рассылка для 
@andreosadchy
 от 
@Люба
Привет!
Напиши, пожалуйста, своему куратору @Люба в личные сообщения адрес электронной почты, который у тебя привязан к аккаунту в Яндекс. Практикуме!
Только проверь это на платформе, прежде чем писать, там может быть что-то другое!
На это сообщение боту не нужно отвечать, пиши прямо @Люба в личные сообщения.
:почтовый_ящик_для_исходящих:
Напоминание про жёсткий дедлайн
ПРИЛОЖЕНИЕ  19:34
Рассылка для 
@andreosadchy
 от 
@Люба
Привет!
Это бот-напоминалка. Отвечать на это сообщение не нужно.
В нашей когорте 11 июня(суббота) будет жёсткий дедлайн, к которому у тебя должны быть сданы=зачтены следующие проекты:
1. Введение в машинное обучение
2. Обучение с учителем
3. Машинное обучение в бизнесе
4. Сборный проект №2
Проверь, что ты всё успеваешь и точно уложишься в срок. При необходимости куратор сможет добавить 2-3 дня к дедлайну, только напиши ей об этом.
Твоего куратора зовут Люба
Если ты понимаешь, что в сроки совершенно не получается уложиться, то тоже напиши куратору. Она тебе расскажет про твои варианты развития событий!
Хороших выходных и мягкого жёсткого дедлайна!:буп_коту:
отвечать на это сообщение не нужно
Новое


Slackbot
  18:19
Practicum Students переходит на новый план Slack, который не поддерживает гостевые аккаунты. Ваш гостевой доступ будет закрыт 16 декабря 2022 г..
Нужна помощь? Свяжитесь с администратором или обратитесь в службу поддержки.











Сообщение Slackbot








Чтобы добавить новую строку, нажмите Shift + ВВОД.
Обсуждение
37_ds_exerciser



df_test = pd.DataFrame([
    [1, 36.5, 5.9, 17.9, 2, 7, 0],
    [2, 71.7, 12.2, 34.3, 5, 21, 1],
    [3, 88.0, 18.1, 58.2, 17, 17, 1],
], columns=columns)

test_features = df_test.drop('кондиционер', axis=1)

def nearest_neighbor_predict(train_features, train_target, new_features):
    distances = []
    for i in range(train_features.shape[0]):
        vector = train_features.loc[i].values
        distances.append(distance.euclidean(new_features, vector))
    best_index = np.array(distances).argmin()
    return train_target.loc[best_index]

class NearestNeighborClassificator:
    def fit(self, features_train, target_train):
        self.features_train = features_train
        self.target_train = target_train
        
    def predict(self, vector):
        new_target = []
        for i in range(len(vector)):
            new_target.append(nearest_neighbor_predict(self.features_train, self.target_train, vector.loc[i]))
        return pd.Series(new_target)
        
model = NearestNeighborClassificator()
model.fit(train_features, train_target)
new_predictions = model.predict(test_features)
print(new_predictions)

Создание матриц
1. Из списка строк создайте матрицу размером 2x3, в первой строке которой указаны 3, 14, 159, а во второй — числа -2, 7, 183.

c

print(matrix) 

2. Мобильный оператор «Шмеляйн» предоставляет клиентам ежедневные данные об обслуживании.
В таблице указано, сколько СМС, минут разговора и мегабайтов потратили в понедельник пять клиентов. 
Преобразуйте таблицу в матрицу и выведите её на печать (уже в прекоде).
import numpy as np
import pandas as pd

monday_df = pd.DataFrame({'Минуты': [10, 3, 15, 27, 7], 
                          'СМС': [2, 5, 3, 0, 1], 
                          'Мбайты': [72, 111, 50, 76, 85]})
 
monday = monday_df.values # < напишите код здесь >

print(monday)

3. Из полученной матрицы выделите данные по клиенту №4 (нумерация начинается с нуля). 
Напечатайте результат на экране.
import numpy as np
import pandas as pd

monday_df = pd.DataFrame({'Минуты': [10, 3, 15, 27, 7], 
                          'СМС': [2, 5, 3, 0, 1], 
                          'Мбайты': [72, 111, 50, 76, 85]}) 
 
monday = monday_df.values

print(monday[3])

4. Из матрицы monday выделите, сколько интернет-трафика потратил каждый клиент. 
Напечатайте результат на экране.
import numpy as np
import pandas as pd

monday_df = pd.DataFrame({'Минуты': [10, 3, 15, 27, 7], 
                          'СМС': [2, 5, 3, 0, 1], 
                          'Мбайты': [72, 111, 50, 76, 85]})
 
monday = monday_df.values

print(monday[:,2])

Операции с элементами матриц
Задача.Мобильный оператор «Шмеляйн» собрал недельную статистику по пяти клиентам.
Постройте матрицу: сколько за неделю клиенты суммарно потратили минут разговора, СМС и интернет-трафика.
Напечатайте её на экране (уже в прекоде).
Постройте прогноз на месяц. По недельным данным найдите, сколько каждый клиент в среднем использует сервисы за день.
Затем умножьте это среднее значение на 30 (количество дней в следующем месяце).
import numpy as np
import pandas as pd

services = ['Минуты', 'СМС', 'Мбайты']

monday = np.array([
    [10, 2, 72],
    [3, 5, 111],
    [15, 3, 50],
    [27, 0, 76],
    [7, 1, 85]])

tuesday = np.array([
    [33, 0, 108],
    [21, 5, 70],
    [15, 2, 15],
    [29, 6, 34],
    [2, 1, 146]])

wednesday = np.array([
    [16, 0, 20],
    [23, 5, 34],
    [5, 0, 159],
    [35, 1, 74],
    [5, 0, 15]])

thursday = np.array([
    [25, 1, 53],
    [15, 0, 26],
    [10, 0, 73],
    [18, 1, 24],
    [2, 2, 24]])

friday = np.array([
    [32, 0, 33],
    [4, 0, 135],
    [2, 2, 21],
    [18, 5, 56],
    [27, 2, 21]])

saturday = np.array([
    [28, 0, 123],
    [16, 5, 165],
    [10, 1, 12],
    [42, 4, 80],
    [18, 2, 20]])

sunday = np.array([
    [30, 4, 243],
    [18, 2, 23],
    [12, 2, 18],
    [23, 0, 65],
    [34, 0, 90]])

weekly = monday + tuesday + wednesday + thursday + friday + saturday + sunday

print('В неделю')
print(pd.DataFrame(weekly, columns=services))
print()

forecast = weekly * 30 / 7

print('Прогноз на месяц')
print(pd.DataFrame(forecast, dtype=int, columns=services))

Умножение матрицы на вектор
Рассчитайте количество минут разговора, СМС и объём интернет-трафика, которые потратил за месяц клиент №1.
Сохраните результат в переменной client_services и напечатайте результат на экране (уже в прекоде).

import numpy as np
import pandas as pd

services = ['Минуты', 'СМС', 'Мбайты']
packs_names = ['«За рулём»', '«В метро»']
packs = np.array([
    [20, 5],
    [2, 5],
    [500, 1000]])

clients_packs = np.array([
    [1, 2],
    [2, 3],
    [4, 1],
    [2, 3],
    [5, 0]])

client = 1
print('Пакеты клиента') 
print(pd.DataFrame(clients_packs[client], index=packs_names, columns=['']))
print()

client_vector = clients_packs[1, :]
client_services = np.dot(packs, client_vector)

print("Минуты, СМС, Мбайты")
print(client_services)

Транспонирование матриц
Задача.Вычислите, сколько понадобится материалов, чтобы произвести 16 столов, 60 стульев и 4 скамьи.
Сохраните результат в переменной materials и напечатайте результат на экране (уже в прекоде).
import numpy  as np
import pandas as pd

materials_names = ['Древесная плита', 'Металлическая труба', 'Винты']

# англ. производство
manufacture = np.array([
    [0.2, 1.2, 8],
    [0.5, 0.8, 6],
    [0.8, 1.6, 8]])

# англ. мебель
furniture = [60, 4, 16]

materials = np.dot(manufacture.T,furniture) 

print(pd.DataFrame([materials], index=[''], columns=materials_names))

Матричное умножение
1. Вернёмся к мобильному оператору «Шмеляйн». 
В вашем распоряжении матрица, которая содержит данные о составе пакетов клиентов за месяц.
Рассчитайте количество минут, СМС и мегабайтов, которые потратили все клиенты за месяц. 
Выведите результат в матрице clients_services, у которой строки соответствуют клиентам, а столбцы — сервисам.
import numpy as np
import pandas as pd

services = ['Минуты', 'СМС', 'Мбайты']
packs_names = ['«За рулём»', '«В метро»']
packs = np.array([
    [20, 5],
    [2, 5],
    [500, 1000]])

clients_packs = np.array([
    [1, 2],
    [2, 3],
    [4, 1],
    [2, 3],
    [5, 0]])

print('Пакеты')
print(pd.DataFrame(clients_packs, columns=packs_names))
print()

clients_services =  np.dot(clients_packs, packs.T)

print('Минуты, СМС и Мбайты')
print(pd.DataFrame(clients_services, columns=services))
print()

2.
Данные сохранили в двух матрицах:
manufacture — расход материалов на разную мебель;
furniture — набор мебели в каждом заведении.
Шаг 1
Посчитайте, сколько материалов уходит на каждое заведение.
Результат сохраните в матрице venues_material. Тогда функция print() выведет его на экран.
Столбцами станут материалы, а строками — типы заведений:
image
Шаг 2
Посчитайте расход материалов, если заказы поступили от 18 кофеен, 12 бистро и 7 ресторанов.
Эти данные мы записали в переменной venues (англ. «заведения»).
Результат сохраните в переменной total_materials. Тогда функция print() выведет его на экран.

import numpy  as np
import pandas as pd

materials_names = ['Древесная плита', 'Металлическая труба', 'Винты']
venues_names = ['Кофейня', 'Бистро', 'Ресторан']

# матрица, где по строкам указана мебель, а по столбцам — материалы для её изготовления
manufacture = np.array([
    [0.2, 1.2, 8],
    [0.5, 0.8, 6],
    [0.8, 1.6, 8]])

# матрица, где по строкам указаны заведения, а по столбцам — виды мебели
furniture = np.array([
    [12, 0, 3],
    [40, 2, 10],
    [60, 6, 18]])

venues_materials =  furniture @ manufacture

print('По заведениям')
print(pd.DataFrame(venues_materials, index=venues_names, columns=materials_names))
print()

venues = np.array([18, 12, 7])

total_materials = np.dot(venues,venues_materials)

print('Всего')
print(pd.DataFrame([total_materials], index=[''], columns=materials_names))

#Обучение линейной регрессии

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

columns = ['комнаты', 'площадь', 'кухня', 'пл. жилая', 'этаж', 'всего этажей', 'цена']

data = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 4200000],
    [1, 38.0, 8.5, 19.2, 9, 17, 3500000],
    [1, 34.7, 10.3, 19.8, 1, 9, 5100000],
    [1, 45.9, 11.1, 17.5, 11, 23, 6300000],
    [1, 42.4, 10.0, 19.9, 6, 14, 5900000],
    [1, 46.0, 10.2, 20.5, 3, 12, 8100000],
    [2, 77.7, 13.2, 39.3, 3, 17, 7400000],
    [2, 69.8, 11.1, 31.4, 12, 23, 7200000],
    [2, 78.2, 19.4, 33.2, 4, 9, 6800000],
    [2, 55.5, 7.8, 29.6, 1, 25, 9300000],
    [2, 74.3, 16.0, 34.2, 14, 17, 10600000],
    [2, 78.3, 12.3, 42.6, 23, 23, 8500000],
    [2, 74.0, 18.1, 49.0, 8, 9, 6000000],
    [2, 91.4, 20.1, 60.4, 2, 10, 7200000],
    [3, 85.0, 17.8, 56.1, 14, 14, 12500000],
    [3, 79.8, 9.8, 44.8, 9, 10, 13200000],
    [3, 72.0, 10.2, 37.3, 7, 9, 15100000],
    [3, 95.3, 11.0, 51.5, 15, 23, 9800000],
    [3, 69.3, 8.5, 39.3, 4, 9, 11400000],
    [3, 89.8, 11.2, 58.2, 24, 25, 16300000],
], columns=columns)

features = data.drop('цена', axis=1)
target = data['цена']

class LinearRegression:
    def fit(self, train_features, train_target):
        self.w = None
        self.w0 = None


    def predict(self, test_features):
        return np.zeros(test_features.shape[0]) 
    

model = LinearRegression()
model.fit(features,target)
predictions = model.predict(features)
print(r2_score(target, predictions))


2. Напишите метод predict() для вычисления предсказания линейной регрессии.
В методе fit() обновите заглушку для параметров w и w0. Вектор w заполните нулями, а в w0 сохраните среднее значение целевого признака.
   
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

columns = ['комнаты', 'площадь', 'кухня', 'пл. жилая', 'этаж', 'всего этажей', 'цена']

data = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 4200000],
    [1, 38.0, 8.5, 19.2, 9, 17, 3500000],
    [1, 34.7, 10.3, 19.8, 1, 9, 5100000],
    [1, 45.9, 11.1, 17.5, 11, 23, 6300000],
    [1, 42.4, 10.0, 19.9, 6, 14, 5900000],
    [1, 46.0, 10.2, 20.5, 3, 12, 8100000],
    [2, 77.7, 13.2, 39.3, 3, 17, 7400000],
    [2, 69.8, 11.1, 31.4, 12, 23, 7200000],
    [2, 78.2, 19.4, 33.2, 4, 9, 6800000],
    [2, 55.5, 7.8, 29.6, 1, 25, 9300000],
    [2, 74.3, 16.0, 34.2, 14, 17, 10600000],
    [2, 78.3, 12.3, 42.6, 23, 23, 8500000],
    [2, 74.0, 18.1, 49.0, 8, 9, 6000000],
    [2, 91.4, 20.1, 60.4, 2, 10, 7200000],
    [3, 85.0, 17.8, 56.1, 14, 14, 12500000],
    [3, 79.8, 9.8, 44.8, 9, 10, 13200000],
    [3, 72.0, 10.2, 37.3, 7, 9, 15100000],
    [3, 95.3, 11.0, 51.5, 15, 23, 9800000],
    [3, 69.3, 8.5, 39.3, 4, 9, 11400000],
    [3, 89.8, 11.2, 58.2, 24, 25, 16300000],
], columns=columns)

features = data.drop('цена', axis=1)
target = data['цена']

class LinearRegression:
    def fit(self, train_features, train_target):
        self.w = np.zeros(train_features.shape[1])
        self.w0 = np.mean(train_target)
    def predict(self, test_features):
        return test_features@self.w+self.w0
    
model = LinearRegression()
model.fit(features, target)
predictions = model.predict(features)
print(r2_score(target, predictions))

3. В прекоде применили сокращённую запись формулы линейной регрессии: в обучающую выборку добавили единичный столбец.
Допишите код для вычисления w по формуле минимизации MSE. Затем вернитесь к исходной записи параметров w и w0 (уже в прекоде).   


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

columns = ['комнаты', 'площадь', 'кухня', 'пл. жилая', 'этаж', 'всего этажей', 'цена']

data = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 4200000],
    [1, 38.0, 8.5, 19.2, 9, 17, 3500000],
    [1, 34.7, 10.3, 19.8, 1, 9, 5100000],
    [1, 45.9, 11.1, 17.5, 11, 23, 6300000],
    [1, 42.4, 10.0, 19.9, 6, 14, 5900000],
    [1, 46.0, 10.2, 20.5, 3, 12, 8100000],
    [2, 77.7, 13.2, 39.3, 3, 17, 7400000],
    [2, 69.8, 11.1, 31.4, 12, 23, 7200000],
    [2, 78.2, 19.4, 33.2, 4, 9, 6800000],
    [2, 55.5, 7.8, 29.6, 1, 25, 9300000],
    [2, 74.3, 16.0, 34.2, 14, 17, 10600000],
    [2, 78.3, 12.3, 42.6, 23, 23, 8500000],
    [2, 74.0, 18.1, 49.0, 8, 9, 6000000],
    [2, 91.4, 20.1, 60.4, 2, 10, 7200000],
    [3, 85.0, 17.8, 56.1, 14, 14, 12500000],
    [3, 79.8, 9.8, 44.8, 9, 10, 13200000],
    [3, 72.0, 10.2, 37.3, 7, 9, 15100000],
    [3, 95.3, 11.0, 51.5, 15, 23, 9800000],
    [3, 69.3, 8.5, 39.3, 4, 9, 11400000],
    [3, 89.8, 11.2, 58.2, 24, 25, 16300000],
], columns=columns)

features = data.drop('цена', axis=1)
target = data['цена']

class LinearRegression:
    def fit(self, train_features, train_target):
        X = np.concatenate((np.ones((train_features.shape[0], 1)), train_features), axis=1)
        y = train_target
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.w = w[1:]
        self.w0 = w[0]

    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0
    
model = LinearRegression()
model.fit(features, target)
predictions = model.predict(features)
print(r2_score(target, predictions))