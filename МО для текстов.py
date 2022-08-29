машинное обучение для текстов
#Лемматизация
Задача.
Напишите функцию lemmatize(text). На вход она принимает текст из корпуса и возвращает лемматизированную строку.
Возьмите первый текст из датасета tweets.csv. Код напечатает на экране исходный и лематизированный тексты.

import pandas as pd
from pymystem3 import Mystem
m = Mystem() 
data = pd.read_csv('/datasets/tweets.csv')
corpus = data['text'].values.astype('U')

def lemmatize(text):
    x = m.lemmatize(text) 
    return "".join(x)

print("Исходный текст:", corpus[0])
print("Лемматизированный текст:", lemmatize(corpus[0]))


#Регулярные выражения

Напишите функцию clear_text(text), которая оставит в тексте только кириллические символы и пробелы.
На вход она принимает текст, а возвращает очищенный текст. Дополнительно уберите лишние пробелы.
Напечатайте на экране исходный текст, а затем очищенный и лемматизированный тексты (уже в прекоде).
import pandas as pd
from pymystem3 import Mystem
import re 

data = pd.read_csv('/datasets/tweets.csv')
corpus = list(data['text'])


def lemmatize(text):
    m = Mystem()
    lemm_list = m.lemmatize(text)
    lemm_text = "".join(lemm_list)
        
    return lemm_text


def clear_text(text):
    clear_text = re.sub(r'[^а-яА-ЯёЁ ]', ' ', text)
    clear_text = " ".join(clear_text.split())
    return clear_text
print("Исходный текст:", corpus[0])
print("Очищенный и лемматизированный текст:", lemmatize(clear_text(corpus[0])))

#Мешок слов и N-граммы

