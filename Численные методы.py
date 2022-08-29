#Вычислительная сложность

Итеративные методы
Допишите код функции bisect(), которая методом бисекции найдёт решение уравнения. На вход она принимает:
function — функция с искомыми нулевыми значениями. В Python аргументами можно передавать функции. А вызывают её так: function(x);
left, right — левый и правый концы отрезка;
error — допустимая величина ошибки (от неё зависит точность алгоритма).
В прекоде уже есть проверка метода на двух функциях.

import math

def bisect(function, left, right, error):
    # Цикл while повторяет код, пока выполняется условие.
    # Добавили в него условие остановки.
    while right - left > error:

        # проверяем, нет ли нулей
        if function(left) == 0:
            return left
        if function(right) == 0:
            return right
        # < напишите код здесь >
        
        # делим отрезок пополам и находим новый отрезок
        middle = (left + right) / 2
        if function(left) * function(middle) < 0:
            right = middle
        if function(right) * function(middle) < 0:
             left = middle
        # < напишите код здесь >
    return left


def f1(x):
    return x**3 - x**2 - 2*x 


def f2(x):
    return (x+1)*math.log10(x) - x**0.75


print(bisect(f1, 1, 4, 0.000001))
print(bisect(f2, 1, 4, 0.000001))

#Градиентный спуск на Python
1. Мы записали функцию f, в коде назвали её func(). Напишите функцию gradient(), которая по формуле вычисляет её градиент. Проверьте эту функцию на нескольких векторах (уже в прекоде).
import numpy as np

def func(x):
    return (x[0] + x[1] - 1)**2 + (x[0] - x[1] - 2)**2

def gradient(x):
    return np.array([4 * x[0] - 6, 4 * x[1] + 2 ])

print(gradient(np.array([0, 0])))
print(gradient(np.array([0.5, 0.3])))

2. Напишите функцию gradient_descent(), реализующую алгоритм градиентного спуска для функции f(x). На вход она принимает:
initialization — начальное значение вектора x;
step_size — размер шага μ;
iterations — количество итераций.
Функция возвращает значение вектора x после заданного числа итераций. Проверьте функцию на разных количествах итераций (уже в прекоде).

import numpy as np

def func(x):
    return (x[0] + x[1] - 1)**2 + (x[0] - x[1] - 2)**2

def gradient(x):
    return np.array([4 * x[0] - 6, 4 * x[1] + 2])

def gradient_descent(initialization, step_size, iterations):
    x = initialization
    for i in range(iterations):
        x = x - step_size * gradient(x)
    return x


print(gradient_descent(np.array([0, 0]), 0.1, 5))
print(gradient_descent(np.array([0, 0]), 0.1, 100))

#SGD на Python

1. Разработку алгоритма SGD начните с заглушки. Загрузка данных и запуск алгоритмов уже в прекоде. Допишите класс модели:
Добавьте в модель гиперпараметры epochs и batch_size. Соблюдая порядок, добавьте их в инициализатор класса и сохраните в атрибутах self.epochs и self.batch_size.
В функции fit() задайте, что начальные веса w равны нулям.
В функции predict() напишите формулу для вычисления предсказания.
Функция print() выведет значения метрики R2 на экран.

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

data_train = pd.read_csv('/datasets/train_data_n.csv')
features_train = data_train.drop(['target'], axis=1)
target_train = data_train['target']

data_test = pd.read_csv('/datasets/test_data_n.csv')
features_test = data_test.drop(['target'], axis=1)
target_test = data_test['target']


class SGDLinearRegression:
    def __init__(self, step_size, epochs, batch_size):
        self.step_size = step_size 
        self.epochs = epochs 
        self.batch_size = batch_size

    def fit(self, train_features, train_target):
        X = np.concatenate((np.ones((train_features.shape[0], 1)), train_features), axis=1)
        y = train_target
        w =  np.zeros(X.shape[1])

        # в следующих задачах вы напишете здесь алгоритм SGD

        self.w = w[1:]
        self.w0 = w[0]

    def predict(self, test_features):
        return features_test.dot(self.w) + self.w0


# мы уже передали подходящие для обучения параметры
model = SGDLinearRegression(0.01, 1, 200)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))


2. Добавьте циклы по эпохам и батчам, учитывая шаг по антиградиенту.
Вам нужно:
    вычислить количество батчей,
    найти градиент по батчу,
    сделать для весов шаг по антиградиенту.
Функция print() выведет значения метрики R2 на экран.


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


data_train = pd.read_csv('/datasets/train_data_n.csv')
features_train = data_train.drop(['target'], axis=1)
target_train = data_train['target']

data_test = pd.read_csv('/datasets/test_data_n.csv')
features_test = data_test.drop(['target'], axis=1)
target_test = data_test['target']


class SGDLinearRegression:
    def __init__(self, step_size, epochs, batch_size):
        self.step_size = step_size
        self.epochs = epochs
        self.batch_size = batch_size
    
    def fit(self, train_features, train_target):
        X = np.concatenate((np.ones((train_features.shape[0], 1)), train_features), axis=1)        
        y = train_target
        w = np.zeros(X.shape[1])
        
        for _ in range(self.epochs):
            batches_count = X.shape[0] //self.batch_size
            for i in range(batches_count):
                begin = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X[begin:end, :]
                y_batch = y[begin:end]
                gradient = 2 * X_batch.T.dot(X_batch.dot(w) - y_batch) / X_batch.shape[0]
                w -= self.step_size * gradient

        self.w = w[1:]
        self.w0 = w[0]
        self.batches_count = batches_count
        
    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0
    
model = SGDLinearRegression(0.01, 10, 100)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))


#Регуляризация линейной регрессии

Добавьте в модель регуляризацию.
Допишите код регуляризации:
приравняйте элемент с индексом ноль в векторе reg к нулю (обычно сдвиг не включается в регуляризацию);
к градиенту функции прибавьте слагаемое для регуляризации.
В код модели из прошлой задачи мы добавили:
новый гиперпараметр — вес регуляризации (reg_weight);
обучение и тестирование с разными значениями.
Напечатайте значения весов регуляризации и метрики R2 на экране (уже в прекоде)


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


data_train = pd.read_csv('/datasets/train_data_n.csv')
features_train = data_train.drop(['target'], axis=1)
target_train = data_train['target']

data_test = pd.read_csv('/datasets/test_data_n.csv')
features_test = data_test.drop(['target'], axis=1)
target_test = data_test['target']


class SGDLinearRegression:
    def __init__(self, step_size, epochs, batch_size, reg_weight):
        self.step_size = step_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_weight = reg_weight
    
    def fit(self, train_features, train_target):
        X = np.concatenate((np.ones((train_features.shape[0], 1)), train_features), axis=1)        
        y = train_target
        w = np.zeros(X.shape[1])
        
        for _ in range(self.epochs):
            batches_count = X.shape[0] // self.batch_size
            for i in range(batches_count):
                begin = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X[begin:end, :]
                y_batch = y[begin:end]
                
                gradient = 2 * X_batch.T.dot(X_batch.dot(w) - y_batch) / X_batch.shape[0]
								# копируем вектор w, чтобы его не менять
                reg = 2 * w.copy()
                # < напишите код здесь >
                reg[0]=reg[0]-reg[0]
                gradient += self.reg_weight * reg # < напишите код здесь >
                
                w -= self.step_size * gradient

        self.w = w[1:]
        self.w0 = w[0]

    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0

# Чтобы сравнить гребневую регрессию с линейной, начнём с
# веса регуляризации, равного 0. Затем добавим
# обучение с его различными значениями.
print("Регуляризация:", 0.0)
model = SGDLinearRegression(0.01, 10, 100, 0.0)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))

print("Регуляризация:", 0.1)
model = SGDLinearRegression(0.01, 10, 100, 0.1)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))

print("Регуляризация:", 1.0)
model = SGDLinearRegression(0.01, 10, 100, 1.0)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))

print("Регуляризация:", 10.0)
model = SGDLinearRegression(0.01, 10, 100, 10.0)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))