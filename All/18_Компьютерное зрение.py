#Библиотека Keras

1.Обучите модель линейной регрессии средствами библиотеки Keras. 
Метод fit() напечатает на экране прогресс обучения и значение ошибки. 
Чтобы формат ответа был понятным, добавьте в этот метод аргумент verbose=2, где 2 означает вывод в консоль. 
Если указать 0, то его не будет вовсе; если — 1, то вывод предназначен для Jupyter Notebook.

import pandas as pd
from sklearn.linear_model import LinearRegression
from tensorflow import keras

data = pd.read_csv('/datasets/train_data_n.csv')
features = data.drop('target', axis=1)
target = data['target']

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=features.shape[1]))
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(features, target,verbose=2) 

2. Подберите такое количество эпох, чтобы получить MSE меньше 6.55. 
Добавить эпохи можно, указав аргумент epochs в методе model.fit().

import pandas as pd
from tensorflow import keras

data = pd.read_csv('/datasets/train_data_n.csv')
features = data.drop('target', axis=1)
target = data['target']

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=features.shape[1]))
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(features, target,epochs = 5 ,verbose = 2)

3. Найдите значение функции потерь на валидационной выборке. 
Передайте модели валидационную выборку в аргументе validation_data метода model.fit(). 
Загрузку валидационной выборки мы сделали (уже в прекоде).

import pandas as pd
from tensorflow import keras

data_train = pd.read_csv('/datasets/train_data_n.csv')
features_train = data_train.drop('target', axis=1)
target_train = data_train['target']

data_valid = pd.read_csv('/datasets/test_data_n.csv')
features_valid = data_valid.drop('target', axis=1)
target_valid = data_valid['target']

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=features_train.shape[1]))
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(features_train, target_train,verbose=2, epochs=5, validation_data = (features_valid,target_valid))

#Логистическая регрессия в Keras


1. Обучите логистическую регрессию на данных, загруженных в прекоде. Найдите значение функции потерь на валидационной выборке.
Установите количество эпох, равным пяти.
Чтобы напечатать прогресс обучения, задайте аргумент verbose=2 в функции fit().

import pandas as pd
from tensorflow import keras

df = pd.read_csv('/datasets/train_data_n.csv')
df['target'] = (df['target'] > df['target'].median()).astype(int)
features_train = df.drop('target', axis=1)
target_train = df['target']

df_val = pd.read_csv('/datasets/test_data_n.csv')
df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
features_valid = df_val.drop('target', axis=1)
target_valid = df_val['target']


model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=features_train.shape[1],
                            activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(features_train, target_train,verbose=2, epochs=5, validation_data = (features_valid,target_valid))


2. Найдите accuracy модели на валидационной выборке. 
Предсказания модели вычисляются функцией predict(), как в sklearn. Сигмоида вернёт числа от 0 до 1, преобразуйте их в классы, сравнив с 0.5.
Напечатайте на экране значение accuracy (уже в прекоде). 
Чтобы выводу не мешал прогресс обучения, задайте verbose=0.


import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score


df = pd.read_csv('/datasets/train_data_n.csv')
df['target'] = (df['target'] > df['target'].median()).astype(int)
features_train = df.drop('target', axis=1)
target_train = df['target']

df_val = pd.read_csv('/datasets/test_data_n.csv')
df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
features_valid = df_val.drop('target', axis=1)
target_valid = df_val['target']

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=features_train.shape[1], 
                             activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(features_train, target_train, epochs=5, verbose=0,
          validation_data=(features_valid, target_valid))


predictions = model.predict(features_valid) > 0.5
score = accuracy_score(target_valid, predictions)
print("Accuracy:", score) 


3. Обучение нейронной сети обычно занимает много времени. Сделайте так, чтобы отследить качество модели можно было на каждой эпохе. 
Для этого добавьте параметр metrics=['acc'] (от англ. accuracy) в методе compile().
Чтобы улучшить значение accuracy, обучите модель на десяти эпохах.

import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score


df = pd.read_csv('/datasets/train_data_n.csv')
df['target'] = (df['target'] > df['target'].median()).astype(int)
features_train = df.drop('target', axis=1)
target_train = df['target']

df_val = pd.read_csv('/datasets/test_data_n.csv')
df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
features_valid = df_val.drop('target', axis=1)
target_valid = df_val['target']

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=features_train.shape[1], 
                             activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['acc'])
model.fit(features_train, target_train, epochs=10, verbose=2,
          validation_data=(features_valid, target_valid))


#Полносвязные нейронные сети в Keras

Задача. Добавьте в нейронную сеть ещё один слой. 
Пусть у первого скрытого слоя будет 10 нейронов units с сигмоидной активацией.
Во втором выходном слое будет один нейрон с сигмоидой: её рассмотрим как вероятность класса «1».
Обучите нейронную сеть на 10 эпохах, напечатав на экране прогресс обучения.

import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score


df = pd.read_csv('/datasets/train_data_n.csv')
df['target'] = (df['target'] > df['target'].median()).astype(int)
features_train = df.drop('target', axis=1)
target_train = df['target']

df_val = pd.read_csv('/datasets/test_data_n.csv')
df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
features_valid = df_val.drop('target', axis=1)
target_valid = df_val['target']

model = keras.models.Sequential()
# первый скрытый слой у него 10 нейронов с сигмоидной активацией
model.add(keras.layers.Dense(units=10, input_dim=features_train.shape[1],
                             activation='sigmoid'))
# во втором выходном слое 1 нейрон с сигмоидой, вероятность класса =1                              
model.add(keras.layers.Dense(units=1, input_dim=1,
                             activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

model.fit(features_train, target_train, epochs=10, verbose=2,
          validation_data=(features_valid, target_valid))


#Работа с изображениями в Python

1. Постройте изображение вызовом функции plt.imshow() (от англ. image show, «показать изображение»). 
Дополнительные аргументы указывать не нужно.
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image = Image.open('/datasets/ds_cv_images/face.png')
array = np.array(image)

plt.imshow(array) 


2. Изучите функцию imshow() и добавьте аргумент, который сделает цветовую гамму чёрно-белой.
Затем добавьте к изображению шкалу цвета вызовом функции colorbar() (англ. «цветовая шкала»): plt.colorbar() 

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('/datasets/ds_cv_images/face.png')
array = np.array(image)

plt.imshow(array, cmap ='gray')
plt.colorbar()

3. Перекрасьте верхний левый угол изображения в чёрный цвет (значение 0), а нижний правый — в белый (255).
С изображением можно работать как с двумерным массивом.

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('/datasets/ds_cv_images/face.png')
array = np.array(image)

array[0, 0] = 0 
array[14, 12] = 255 

plt.imshow(array, cmap='gray')
plt.colorbar()

4. Чтобы нейронные сети обучались лучше, обычно на вход им передают изображения в диапазоне от 0 до 1.
Приведите масштаб [0, 255] к [0, 1]. Для этого поделите все значения двумерного массива на 255
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('/datasets/ds_cv_images/face.png')
array = np.array(image)

array = array / 255. 

plt.imshow(array, cmap='gray')
plt.colorbar()

#Цветные изображения


1. Убедитесь, что каналы хранятся в третьей координате. Для этого выведите размер массива, полученного из картинки.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open('/datasets/ds_cv_images/cat.jpg')
array = np.array(image)

print(array.shape)
#(300, 400, 3)

2. Изобразите на экране канал только с красным цветом.
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open('/datasets/ds_cv_images/cat.jpg')
array = np.array(image)

red_channel =array[:,:,0]

plt.imshow(red_channel)
plt.colorbar()

#Классификация изображений
1. Посмотрите на данные. Мы уже загрузили выборки, разбили их на признаки и целевой признак.
Выведите на экран размеры признаков в обеих выборках.
Затем напечатайте значение целевого признака первого изображения из обучающей выборки. Само изображение выведите в чёрно-белом цвете (уже в прекоде).

import matplotlib.pyplot as plt
import numpy as np


features_train = np.load('/datasets/fashion_mnist/train_features.npy')
target_train = np.load('/datasets/fashion_mnist/train_target.npy')
features_test = np.load('/datasets/fashion_mnist/test_features.npy')
target_test = np.load('/datasets/fashion_mnist/test_target.npy')

print("Обучающая:", features_train.shape)
print("Тестовая:", features_test.shape)

print("Класс первого изображения:", target_train[0])
plt.imshow(features_train[0], cmap='gray')

2. В полносвязных сетях подаваемые на вход объекты должны быть строками таблицы, а весь датасет — двумерной таблицей.
Чтобы не было ошибки, преобразуйте датасет из трёхмерного массива в двумерную таблицу. Для этого вам понадобится метод np.array.reshape() (от англ. «изменить размер»).
Преобразуйте features_train так, чтобы в первом значении features_train.shape было количество объектов, а во втором — количество пикселей в изображении.
Таким же способом измените features_test. Напечатайте на экране новые размеры массивов (уже в прекоде).

import matplotlib.pyplot as plt
import numpy as np


features_train = np.load('/datasets/fashion_mnist/train_features.npy')
target_train = np.load('/datasets/fashion_mnist/train_target.npy')
features_test = np.load('/datasets/fashion_mnist/test_features.npy')
target_test = np.load('/datasets/fashion_mnist/test_target.npy')

features_train = features_train.reshape(features_train.shape[0], 28 * 28) 
features_test = features_test.reshape(features_test.shape[0], 28 * 28) 


print("Обучающая:", features_train.shape)
print("Тестовая:", features_test.shape)


3. Постройте и обучите нейронную сеть. Начните с простого — создайте модель логистической регрессии с десятью классами в Keras.
Вам понадобится:
Функция активации 'softmax';
Функция потерь 'sparse_categorical_crossentropy' (англ. «разрежённая категориальная кросс-энтропия»).
Слово sparse говорит о способе кодирования ответов. В задаче требуется просто номер класса, поэтому выбор пал на эту функцию потерь.
Когда ответы кодируются One-Hot-Encoding и классу 9 соответствует целый вектор [0, 0, 0, 0, 0, 0, 0, 0, 1], применяют categorical_crossentropy().
Обучите сеть на одной эпохе. Напечатайте прогресс обучения и значения точности на обучающей и тестовой выборках.

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


features_train = np.load('/datasets/fashion_mnist/train_features.npy')
target_train = np.load('/datasets/fashion_mnist/train_target.npy')
features_test = np.load('/datasets/fashion_mnist/test_features.npy')
target_test = np.load('/datasets/fashion_mnist/test_target.npy')

features_train = features_train.reshape(features_train.shape[0], 28 * 28)
features_test = features_test.reshape(features_test.shape[0], 28 * 28)

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=10, input_dim=28*28,
                             activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['acc'])
model.fit(features_train, target_train, epochs=1, verbose=2,
          validation_data=(features_test, target_test))

