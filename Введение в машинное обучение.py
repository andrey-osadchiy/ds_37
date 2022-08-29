#Классификация и регрессия

1. Преобразуйте исходную задачу в задачу классификации. Создайте новый признак price_class (англ. «ценовой класс»). Чтобы установить его значение, сравните цену с 5 650 000:
если цена больше, значение price_class — единица, 1.0;
если цена меньше или равна, значение price_class — ноль, 0.0.
Функция print() выведет на экран первые пять строк.

import pandas as pd

df = pd.read_csv('/datasets/train_data.csv')

df['price_class'] = 0.0
df.loc[df['last_price']>5650000,'price_class'] = 1.0
print(df.head())


2. Многие библиотеки машинного обучения требуют, чтобы признаки были сохранены в отдельных переменных. Объявите две переменные:
features (англ. «признаки») — запишите в неё признаки;
target (англ. «цель») — целевой признак.
Выведите на экран размеры этих переменных (уже в прекоде).

import pandas as pd

df = pd.read_csv('/datasets/train_data.csv')

df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0


features = df.drop(['last_price'], axis=1)
features = features.drop(['price_class'], axis=1)
target = df['price_class']

print(features.shape)
print(target.shape)

#Библиотека scikit-learn

1. Начните с обучения модели. В пятом уроке вы сохранили обучающий набор данных в переменных features и target.
Чтобы запустить обучение, вызовите метод fit() (англ. «подогнать») и передайте ему как параметр данные.

model.fit(features, target) 

Допишите код. Выведите значение переменной на экран (уже сделано в прекоде).


import pandas as pd
# импортируйте решающее дерево из библиотеки sklearn
from sklearn.tree import DecisionTreeClassifier 

df = pd.read_csv('/datasets/train_data.csv')

df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

# создайте пустую модель и сохраните её в переменной
model = DecisionTreeClassifier()

# обучите модель вызовом метода fit()
model.fit(features, target) 

print(model)


2. Теперь в переменной model полноценная модель.
Чтобы предсказать ответы, нужно вызвать метод predict() (англ. «предсказать») и передать ему таблицу с признаками новых объектов.

    answers = model.predict(new_features) 

Создайте два новых объекта и посмотрите на результаты предсказаний. Объекты задачи — это дома. Выпишите значения признаков каждого:
В первой квартире целых 12 комнат общей площадью 900 кв. м. Из них жилых 409.7 кв. м, а кухня — 112 кв. м.
Во второй квартире 2 комнаты общей площадью 109 кв. м. Из них жилых 32 кв. м, а кухня — 40.5 кв. м.
Остальные признаки не различаются или разнятся не так сильно. Их мы уже заполнили в прекоде. 
Предскажите ответ и напечатайте результат на экране.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/datasets/train_data.csv')

df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier()

model.fit(features, target)

new_features = pd.DataFrame(
    [[None, None, 2.8, 25, None, 25, 0, 0, 0, None, 0, 30706.0, 7877.0],
     [None, None, 2.75, 25, None, 25, 0, 0, 0, None, 0, 36421.0, 9176.0]],
    columns=features.columns)

# дозаполните таблицу с новыми признаками
new_features.loc[0, 'total_area'] = 900.0
new_features.loc[1, 'total_area'] = 109.0
new_features.loc[0, 'rooms'] = 12
new_features.loc[1, 'rooms'] = 2
new_features.loc[0, 'living_area'] = 409.7
new_features.loc[1, 'living_area'] = 32.0
new_features.loc[0, 'kitchen_area'] = 112.0
new_features.loc[1, 'kitchen_area'] = 40.5

# < напишите код здесь >

# предскажите ответы и напечатайте результат на экране
answers = model.predict(new_features) 
print(answers)
#[1. 0.]
# то есть модеь говорит что 12 комнатная квартира - дорогая а двушка- дешевая.



#Тестовый набор данных
1. Загрузите в переменную test_df три первых объекта из тестовой выборки.
Подготовьте их к классификации: сохраните признаки в переменной test_features (англ. «признаки для теста»), а целевой признак — в test_target (англ. «цель теста»).
Предскажите ответы.
Предсказания сохраните в переменной test_predictions. Напечатайте на экране сначала предсказания, затем — правильные ответы в таком формате:
Предсказания: [... ... ...]
Правильные ответы: [... ... ...] 
Переводить переменные из np.array в list не нужно.
Выясните, много ли ошибок допустила модель?

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# обучающая выборка находится в файле train_data.csv
df = pd.read_csv('/datasets/train_data.csv')

df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345)

model.fit(features, target)

test_df = pd.read_csv('/datasets/test_data.csv')
test_df = test_df.loc[:2]
test_df.loc[test_df['last_price'] > 5650000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 5650000, 'price_class'] = 0
test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target  = test_df['price_class']

test_predictions = model.predict(test_features)
print('Предсказания:',test_predictions)
print('Правильные ответы:', test_target.values)
# < напишите код здесь >

2. Трёх примеров недостаточно, чтобы понять, хорошо или плохо работает модель. Посчитайте количество ошибок модели на всей тестовой выборке.
Напишите функцию error_count()(англ. «подсчёт ошибок»), которая:
Принимает на вход правильные ответы и предсказания модели.
Сравнивает их в цикле for.
Возвращает количество расхождений между ними.
Выведите результат работы error_count() с тестовым набором в таком формате (уже сделано в прекоде):
Ошибок: ... 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/datasets/train_data.csv')

df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1) 
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345)

model.fit(features, target)

test_df = pd.read_csv('/datasets/test_data.csv')

test_df.loc[test_df['last_price'] > 5650000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 5650000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']
test_predictions = model.predict(test_features)

def error_count(answers, predictions):
#    counter = 0
#    for i in range(0,len(answers)):
#        if answers[i] != predictions[i]:
#            counter =+ 1
    return (answers != predictions).sum()
print("Ошибок:", error_count(test_target, test_predictions))


Задача

Напишите функцию accuracy(), которая:
Принимает на вход правильные ответы и предсказания,
Сравнивает их в цикле for.
Возвращает долю правильных ответов.
Получится функция, похожая на error_count().

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/datasets/train_data.csv')

df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345)

model.fit(features, target)

test_df = pd.read_csv('/datasets/test_data.csv')

test_df.loc[test_df['last_price'] > 5650000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 5650000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']
test_predictions = model.predict(test_features)

def error_count(answers, predictions):
    count = 0
    for i in range(len(answers)):
        if answers[i] != predictions[i]:
            count += 1
    return count

def accuracy(answers, predictions):
    right_answers = 0
    for i in range(len(answers)):
        if answers[i] == predictions[i]:
            right_answers += 1
    return right_answers/len(answers)
    

print("Accuracy:", accuracy(test_target, test_predictions)) 


Метрики качества в библиотеке sklearn

Задача
Отличается ли accuracy на обучающей и тестовой выборках? Посчитайте значения и напечатайте на экране результаты в таком формате:
Accuracy
Обучающая выборка: ...
Тестовая выборка: ... 
Предсказания на обучающей выборке сохраните в переменной train_predictions, а на тестовой — в test_predictions.


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

df = pd.read_csv('/datasets/train_data.csv')

df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345)

model.fit(features, target)

train_predictions = model.predict(features)
train_accuracy = accuracy_score(target, train_predictions) 

test_df = pd.read_csv('/datasets/test_data.csv')

test_df.loc[test_df['last_price'] > 5650000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 5650000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']
test_predictions = model.predict(test_features)
test_accuracy = accuracy_score(test_target, test_predictions)
# < напишите здесь код расчёта на обучающей выборке >
# < напишите здесь код расчёта на тестовой выборке >

print("Accuracy")
print("Обучающая выборка:", train_accuracy)
print("Тестовая выборка:", test_accuracy)

Эксперименты с решающим деревом

Научите программу пробовать разные значения одного параметра — максимальной глубины, max_depth.
Алгоритм должен:
    перебрать значения от 1 до 5,
    сохранить модель с лучшим значением метрики accuracy на тренировочном датасете.
    

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('/datasets/train_data.csv')

df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

best_model = None
best_result = 0
for depth in range(1, 6):
	model = DecisionTreeClassifier(random_state=12345, max_depth = depth) # обучите модель с заданной глубиной дерева
	model.fit(features,target) # обучите модель
	predictions = model.predict(features) # получите предсказания модели
	result = accuracy_score(target, predictions) # посчитайте качество модели
	if result > best_result:
		best_model = model
		best_result = result
        
print("Accuracy лучшей модели:", best_result)    


#Деление на две выборки
Задача
Разделите набор данных на обучающую (df_train) и валидационную (df_valid) выборки. 
В валидационной выборке — 25% исходных данных. В random_state положите значение 12345.
В коде мы объявили переменные с признаками для обучения: target_train (англ. цель для обучения) и  features_train (англ. признаки для обучения). Создайте аналогичные переменные для проверки:
    с целевым признаком — target_valid;
    с остальными признаками — features_valid.
Выведите на экран размеры таблиц, которые хранятся в четырёх переменных (сделано в прекоде).

import pandas as pd
# < импортируйте функцию train_test_split из библиотеки sklearn >
from sklearn.model_selection import train_test_split

df = pd.read_csv('/datasets/train_data.csv')
df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

# < разделите данные на обучающую и валидационную выборки >
df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345)

# < создайте переменные для признаков и целевого признака >
features_train = df_train.drop(['last_price', 'price_class'], axis=1)
target_train = df_train['price_class']
features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)# < напишите код здесь >
target_valid = df_valid['price_class']# < напишите код здесь >

print(features_train.shape)
print(target_train.shape)
print(features_valid.shape)
print(target_valid.shape)

#Смена гиперпараметров
Задача
Поменяйте гиперпараметр max_depth от 1 до 5 в цикле. 
Для каждого значения напечатайте на экране качество на валидационной выборке. Формат вывода сделайте таким:
max_depth = 1 : ...
max_depth = 2 : ...
...
max_depth = 5 : ... 
Проверять на тестовой выборке пока не нужно, сначала найдите наилучшую модель.


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('/datasets/train_data.csv')
df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345)

features_train = df_train.drop(['last_price', 'price_class'], axis=1)
target_train = df_train['price_class']
features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)
target_valid = df_valid['price_class']

# < сделайте цикл для max_depth от 1 до 5 >
for depth in range(1, 6):
    model = DecisionTreeClassifier(random_state=12345, max_depth = depth) # обучите модель с заданной глубиной дерева
    model.fit(features_train,target_train) # обучите модель
    predictions_valid = model.predict(features_valid) # получите предсказания модели
    result = accuracy_score(target_valid, predictions_valid) # посчитайте качество модели
    print('max_depth =', depth, ':', result)

#Новые модели: случайный лес

Задача
Обучите модели случайного леса с числом деревьев от 1 до 10. Для этого:
    разделите тренировочную и валидационную выборки,
    при инициализации модели укажите число деревьев равное состоянию счётчика циклов — est;
    обучите модель на тренировочной выборке,
    сохраните модель с лучшим значением accuracy на валидационной выборке.
Качество на тестовой выборке должно получиться не меньше 0.88.    

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('/datasets/train_data.csv')
df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345) # отделите 25% данных для валидационной выборки

features_train = df_train.drop(['last_price', 'price_class'], axis=1)
target_train = df_train['price_class']
features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)
target_valid = df_valid['price_class']

best_model = None
best_result = 0
for est in range(1, 11):
    model = RandomForestClassifier(random_state=12345, n_estimators = est) # обучите модель с заданным количеством деревьев
    model.fit(features_train,target_train) # обучите модель на тренировочной выборке
    result = model.score(features_valid, target_valid)  # посчитайте качество модели на валидационной выборке
    if result > best_result:
        best_model = model
        best_result = result

print("Accuracy наилучшей модели на валидационной выборке:", best_result)


#Логистическая регрессия

Обучите модель логистической регрессии и загрузите её в тренажёр.

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('/datasets/train_data.csv')
df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class'] 

model = LogisticRegression(random_state=12345, solver='lbfgs', max_iter=1000) 
model.fit(features, target) 

# Напишите код здесь

dump(model, 'model_9_1.joblib')


#Сравнение моделей
Name	Качество	Скорость
Решающее дерево	Низкое	Высокая
Случайный лес	Высокое	Низкая
Логистическая регрессия	Среднее	Высокая

#Переходим к регрессии
#Расчёт MSE


1. Напишите функцию mse(). На вход она принимает правильные ответы и предсказания, а возвращает значение средней квадратичной ошибки.
Мы переписали из таблицы значения ответов (реальных затрат) и предсказаний. Выведите на экран MSE.

def mse(answers, predictions):
    e = 0
    for i in range(len(answers)):
        e += ((answers[i] - predictions[i])**2)
    return e/len(answers)             
    
answers = [623, 253, 150, 237]
predictions = [649, 253, 370, 148]

print(mse(answers, predictions))

2. Функция расчёта MSE есть и в sklearn.
Найдите в документации и импортируйте mean_squared_error, чтобы решить ту же задачу. Напечатайте на экране значение MSE.

from sklearn.metrics import mean_squared_error

answers = [623, 253, 150, 237]
predictions = [649, 253, 370, 148]

result = mean_squared_error(answers, predictions)
print(result)


#Интерпретация MSE
1. Подготовьте набор данных и найдите среднее значение цены:
Создайте переменную features со всеми признаками, кроме last_price.
Создайте переменную target с целевым признаком last_price. Поделите его значение на 1 миллион.
Посчитайте среднее значение по элементам переменной target.
Напечатайте результат в таком формате:
    Средняя цена: ...
    
    
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy

df = pd.read_csv('/datasets/train_data.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000 
avg = numpy.mean(target)
# < найдите и напечатайте среднее >\
print('Средняя цена:',avg)    


2. Найдите MSE по обучающей выборке, чтобы средней ценой предсказать ответ. Предсказания запишите в переменную predictions.
Результат напечатайте в таком формате:
    MSE: ... 
Функция mean_squared_error в sklearn довольно капризная. Придётся поработать с документацией или Stack Overflow.

import pandas as pd
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000
#Функция mean_squared_error принимает только последовательности. Создать последовательность из средних значений можно так:
predictions = pd.Series(target.mean(), index=target.index)

mse = mean_squared_error(target, predictions)

print("MSE:", mse)


3. «Квадратные рубли» ни к чему. Чтобы метрика показывала просто рубли, возьмите корень от MSE.
Это величина RMSE (англ. root mean squared error, «корень из средней квадратичной ошибки»). 
Напечатайте результат на экране в таком формате:
RMSE: ... 
Корень из числа извлеките операцией возведения в степень **. Число возведите в степень 0.5. Например:
print(25 ** 0.5) 


import pandas as pd
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000

predictions = pd.Series(target.mean(), index=target.index)
mse = mean_squared_error(target, predictions)
# < извлеките корень из MSE >
rmse = mse ** 0.5
print("RMSE:", rmse)


#Дерево решений в регрессии

Задача
Выделите 25% данных для валидационной выборки, остальные — для обучающей.
Обучите модели дерева решений для задачи регрессии с различными значениями глубины от 1 до 5.
Для каждой модели посчитайте значение метрики RMSE на валидационной выборке.
Сохраните модель с наилучшим значением RMSE на валидационной выборке в переменной best_model.


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000

features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345) # отделите 25% данных для валидационной выборки

best_model = None
best_result = 10000
best_depth = 0
for depth in range(1, 6):
    model = DecisionTreeRegressor(random_state=12345,max_depth=depth) 
    # инициализируйте модель DecisionTreeRegressor с параметром random_state=12345 и max_depth=depth
    model.fit(features_train,target_train) # обучите модель на тренировочной выборке
    predictions_valid = model.predict(features_valid) # получите предсказания модели на валидационной выборке
    result = mean_squared_error(target_valid, predictions_valid)**0.5 
    # посчитайте значение метрики rmse на валидационной выборке
    if result < best_result:
        best_model = model
        best_result = result
        best_depth = depth

print("RMSE наилучшей модели на валидационной выборке:", best_result, "Глубина дерева:", best_depth)


#Случайный лес в регрессии
Задача
Извлеките признаки для обучения в features, а целевой признак last_price поделите на 1000000 и сохраните в переменную target.
Выделите 25% данных для валидационной (тестовой) выборки, остальные — для обучающей.
Обучите модели случайного леса для задачи регрессии:
с количеством деревьев: от 10 до 50 с шагом 10,
с максимальной глубиной от 1 до 10.
Для каждой модели посчитайте RMSE на валидационной выборке.
Сохраните модель с наилучшим значением RMSE на валидационной выборке в переменной best_model.
Код может выполняться около минуты. Это нормально, ведь вы обучаете 50 моделей.


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000

features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

best_model = None
best_result = 10000
best_est = 0
best_depth = 0
for est in range(10, 51, 10):
    for depth in range (1, 11):
        model = RandomForestRegressor(random_state=12345, n_estimators=est,max_depth=depth) 
        # инициализируйте модель RandomForestRegressor с параметрами random_state=12345, n_estimators=est и max_depth=depth
        model.fit(features_train,target_train) # обучите модель на тренировочной выборке
        predictions_valid = model.predict(features_valid) # получите предсказания модели на валидационной выборке
        result = mean_squared_error(target_valid, predictions_valid)**0.5 
        if result < best_result:
            best_model = model
            best_result = result
            best_est = est
            best_depth = depth

print("RMSE наилучшей модели на валидационной выборке:", best_result, "Количество деревьев:", best_est, "Максимальная глубина:", depth)

#Линейная регрессия
Задача
Извлеките признаки для обучения в переменную features и целевой признак last_price в переменную target. Поделите значение целевого признака на 1000000.
Инициализируйте модель линейной регрессии, обучите её. Посчитайте значение метрики RMSE на валидационной выборке и сохраните в переменной result.


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000

features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345) # отделите 25% данных для валидационной выборки

model = LinearRegression() # инициализируйте модель LinearRegression
model.fit(features_train,target_train) # обучите модель на тренировочной выборке
predictions_valid = model.predict(features_valid) # получите предсказания модели на валидационной выборке

result = mean_squared_error(target_valid, predictions_valid)**0.5 
# посчитайте значение метрики RMSE на валидационной выборке
print("RMSE модели линейной регрессии на валидационной выборке:", result)


#Выбираем лучшую модель
Найдите модель, у которой на тестовой выборке RMSE не больше 7.3:
Извлеките целевой признак в переменную target, а остальные — в features.
Отделите 25% данных для валидационной выборки. Для этого передайте train_test_split признаки и значение аргумента test_size.
Инициируйте ту модель, которая показала лучшее значение RMSE в прошлых уроках.
С помощью метода .fit() обучите модель на тренировочной выборке.
Получите предсказания модели на валидационной выборке методом .predict().
Посчитайте значение метрики RMSE на валидационной выборке и сохраните его в переменную result.
