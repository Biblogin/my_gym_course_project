# # src/data_loaders.py Модуль для загрузки и чистки данных
import pandas as pd
from sklearn.model_selection import train_test_split


def get_prepared_data(file_path='data/data.csv'):
    df = pd.read_csv(file_path)

    # Отделяем целевую переменную (Y) от признаков (X)
    y = df['number_people']

    # Удаляем целевую переменную из матрицы признаков,
    # а также удаляем столбец 'date', так как это текст, а линейная регрессия ест только числа.
    X = df.drop(columns=['number_people', 'date'])

    # Прячем 20% данных для финальной проверки
    return train_test_split(X, y, test_size=0.2, random_state=42)