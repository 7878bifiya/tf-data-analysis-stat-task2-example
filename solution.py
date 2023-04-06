import numpy as np
import pandas as pd
import scipy.stats as sps

chat_id = 541133397 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    # расчет пути при каждом измерении и вычисление скоростей
    speeds = np.diff(x) / 20

    # расчет ускорений
    accelerations = np.diff(speeds)

    # расчет среднего и стандартного отклонения
    mean_acceleration = np.mean(accelerations)
    std_error = sps.sem(accelerations)

    # вычисление квантилей распределения Стьюдента
    degrees_of_freedom = len(accelerations) - 1
    t_value = sps.t.ppf((1 + p) / 2, degrees_of_freedom)

    # вычисление границ доверительного интервала
    left_boundary = mean_acceleration - t_value * std_error
    right_boundary = mean_acceleration + t_value * std_error

    # доверительный интервал не может быть отрицательным
    if left_boundary < 0:
        left_boundary = 0

    if right_boundary < 0:
        right_boundary = 0

    return  [left_boundary, right_boundary]
