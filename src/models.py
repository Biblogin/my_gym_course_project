# src/models.py
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def get_linear_regression():
    print("\nОбучаем модель Линейной Регрессии...")
    # модель под капотом будет использовать матричную формулу w^* = (X^T X)^{-1} * X^T * y
    return LinearRegression()


def get_sgd_regression():
    print("\nОбучаем модель через Стохастический Градиентный Спуск (SGD)...")
    # make_pipeline объединяет масштабирование и саму модель в один понятный объект
    # Теперь данные внутри будут автоматически нормализоваться перед каждым шагом спуска
    return make_pipeline(StandardScaler(), SGDRegressor(random_state=42))


def get_kernel_regression(X_sample, y_sample):
    print("\nНачинаем обучение модели через Ядерную Регрессию. Это может занять некоторое время...")

    scaler_sample = StandardScaler()
    X_sample_scaled = scaler_sample.fit_transform(X_sample)

    param_grid = {
        'gamma': [0.4, 0.55, 0.7, 0.85, 1],  # Различная ширина окна
        'alpha': [0.155, 0.17, 0.185, 0.2]  # Сила регуляризации (штраф за сложность)
    }

    print("\tЗапуск GridSearchCV на подвыборке 5000 строк...")
    grid_search = GridSearchCV(
        KernelRidge(kernel='rbf'),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_sample_scaled, y_sample)

    best_gamma = grid_search.best_params_['gamma']
    best_alpha = grid_search.best_params_['alpha']

    print("\t--- Лучшие параметры найдены ---")
    print(f"\tОптимальная gamma: {best_gamma}")
    print(f"\tОптимальная alpha: {best_alpha}\n")

    return make_pipeline(
        StandardScaler(),
        Nystroem(kernel='rbf', gamma=best_gamma, n_components=500, random_state=42),
        Ridge(alpha=best_alpha)
    )


def get_random_forest(X_sample, y_sample):
    print("\nНачинаем обучение модели через Случайный лес...")

    # max_depth ограничивает рост дерева (от переобучения)
    # min_samples_split - сколько минимум объектов должно быть в листе, чтобы его поделить
    param_grid = {
        'max_depth': [15, 20, 50, None],
        'min_samples_split': [2, 10]
    }

    base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    print("\tЗапуск GridSearchCV на подвыборке")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error'
    )

    grid_search.fit(X_sample, y_sample)

    best_depth = grid_search.best_params_['max_depth']
    best_split = grid_search.best_params_['min_samples_split']

    print("\t--- Лучшие параметры найдены ---")
    print(f"\tМаксимальная глубина (max_depth): {best_depth}")
    print(f"\tМин. объектов для разделения (min_samples_split): {best_split}\n")

    return RandomForestRegressor(
        n_estimators=100,
        max_depth=best_depth,
        min_samples_split=best_split,
        random_state=42,
        n_jobs=-1
    )