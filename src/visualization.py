# src/visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_model_behavior_over_time(models_dict, X_train, y_train):
    print("\nГенерация графиков визуализации...")

    # 1. Создаем "синтетический день" (24 часа)
    synthetic_day = pd.DataFrame(columns=X_train.columns)

    # Фиксируем все признаки на их среднем значении
    for col in X_train.columns:
        synthetic_day[col] = [X_train[col].mean()] * 24

    # Меняем только час суток (от 0 до 23)
    synthetic_day['hour'] = np.arange(24)

    # Создаем холст для 4 графиков
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Для фона выберем случайные 500 реальных точек
    sample_X = X_train.sample(n=min(500, len(X_train)), random_state=42)
    sample_y = y_train.loc[sample_X.index]

    # Строим график для каждой обученной модели
    for i, (name, model) in enumerate(models_dict.items()):
        ax = axes[i]

        # Рисуем серым цветом реальные точки (фон)
        ax.scatter(sample_X['hour'], sample_y, color='gray', alpha=0.3, label='Реальные данные')

        # Делаем предсказание нашей модели для синтетического дня
        predictions = model.predict(synthetic_day)

        # Рисуем красным цветом линию/ступеньки предсказания модели
        # Для леса используем step-формат отрисовки, для остальных - обычную линию
        if "Forest" in name:
            ax.plot(synthetic_day['hour'], predictions, color='red', linewidth=3, drawstyle='steps-mid',
                    label='Прогноз модели (Ступени)')
        else:
            ax.plot(synthetic_day['hour'], predictions, color='red', linewidth=3, label='Прогноз модели (Линия)')

        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Час суток (hour)', fontsize=12)
        ax.set_ylabel('Количество людей', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 2))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    plt.tight_layout()
    # Сохраняем картинку
    plt.savefig('models_visualization.png', dpi=300)
    plt.show()