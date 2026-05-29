# main.py
import time
from src.data_loader import get_prepared_data
from src.models import get_linear_regression, get_sgd_regression, get_kernel_regression, get_random_forest
from src.visualization import plot_model_behavior_over_time
from sklearn.metrics import mean_absolute_error, r2_score


def run_experiment(model_name, model_object, X_train, X_test, y_train, y_test):
    # print(f"\nЗапуск модели: {model_name}...")

    # обучаем модель, высчитывая время
    start_time = time.time()
    model_object.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Оценка
    predictions = model_object.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {"Время (сек)": training_time, "MAE": mae, "R2": r2}


if __name__ == "__main__":
    # Загружаем данные
    X_train, X_test, y_train, y_test = get_prepared_data()

    # готовим подвыборку для тюнинга ядерной регрессии
    # Берутся 5000 строк из тренировочного набора данных
    X_sample = X_train.iloc[:5000]
    y_sample = y_train.iloc[:5000]

    # Словарь моделей для сравнения
    models_to_test = {
        "Linear Regression (Analytic)": get_linear_regression(),
        "Linear Regression (SGD)": get_sgd_regression(),
        "Kernel Regression (RBF)": get_kernel_regression(X_sample, y_sample),
        "Random Forest": get_random_forest(X_sample, y_sample)
    }

    results = {}
    for name, model in models_to_test.items():
        results[name] = run_experiment(name, model, X_train, X_test, y_train, y_test)

    # Выводим красивую финальную таблицу результатов
    import pandas as pd

    res_df = pd.DataFrame(results).T
    print("\nФИНАЛЬНОЕ СРАВНЕНИЕ МОДЕЛЕЙ:")
    print(res_df)

    # Отрисовка графиков
    # Модели в словаре models_to_test уже обучены в цикле выше
    plot_model_behavior_over_time(models_to_test, X_train, y_train)